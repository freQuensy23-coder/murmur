#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PKG = 'app.murmur'
TEST = PKG + '.test'
TARGET = PKG + '.testtarget'
RUNNER = TEST + '/app.murmur.shell.MurmurTestRunner'
TEST_CLASS = PKG + '.e2e.StreamAppTest#fixtureToEditors'
CAPS = {'launch': 105_000_000, 'insertion': 140_000_000, 'ocr': 390_000_000, 'full': 910_000_000}
ENV = dict(os.environ)
for k in ('ADB_SERVER_SOCKET', 'ANDROID_ADB_SERVER_ADDRESS', 'ANDROID_ADB_SERVER_PORT', 'ADB_SERVER_PORT', 'ANDROID_SERIAL'):
    ENV.pop(k, None)
ENV["ADB_MDNS_AUTO_CONNECT"] = "0"


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def artifact(directory, entry):
    path = (directory / entry['path']).resolve()
    if not path.is_relative_to(directory):
        raise ValueError('artifact must be inside staging directory')
    if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
        raise RuntimeError(f'payload integrity check failed: {path.name}')
    return path


def inputs(args):
    manifest = json.loads((args.directory / 'manifest.json').read_text())
    apks = [(pkg, manifest['apks'][key]) for pkg, key in [(PKG, 'app'), (TEST, 'test'), (TARGET, 'target')]
            if args.mode != 'launch' or key == 'app']
    required = 5 if args.mode == 'full' else 4 if args.mode == 'ocr' else 0
    models = manifest['models'][:required]
    if len(models) != required:
        raise ValueError('staged model inventory does not cover the requested mode')
    for entry in models:
        if not re.fullmatch(r'[A-Za-z0-9_.@-]+', entry['id']) or entry['id'].startswith('.'):
            raise ValueError('invalid staged model identifier')
        if any(not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', name) for name in entry['files']):
            raise ValueError('invalid staged model filename')
    for entry in [entry for _, entry in apks] + models:
        artifact(args.directory, entry)
    return apks, models


class DeviceRun:
    def __init__(self, args, out):
        sdk = Path(os.environ.get('ANDROID_HOME', Path.home() / '.local/share/android-sdk'))
        self.adb = [os.environ.get('ADB', str(sdk / 'platform-tools/adb')), '-s', args.serial]
        self.deadline = time.monotonic() + args.seconds
        self.out = out
        self.args = args
        self.settings = []
        self.changed = False
        self.pending = None
        self.spent = 0
        self.result = {'serial': args.serial, 'mode': args.mode, 'success': False, 'payload_cap': args.max_bytes,
                       'seconds_cap': args.seconds, 'audio': 'fixture, not microphone'}

    def call(self, *argv, source=None, budget=15, allow=False, cleanup=False):
        remaining = self.deadline - time.monotonic() - (0 if cleanup else 8)
        if remaining <= 0:
            raise TimeoutError('overall deadline reached')
        r = subprocess.run([*self.adb, *argv], stdin=source if source is not None else subprocess.DEVNULL,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=ENV,
                           timeout=min(budget, remaining))
        output = r.stdout.decode('utf-8', errors='replace').strip()
        if r.returncode and not allow:
            raise RuntimeError(f'adb {argv[0]}: {output[-1000:]}')
        return output

    def shell(self, *argv, **kwargs):
        return self.call('shell', '-T', shlex.join(argv), **kwargs)

    def as_app(self, script, **kwargs):
        return self.shell('run-as', PKG, 'sh', '-c', script, **kwargs)

    def guard(self):
        if self.call('get-state') != 'device':
            raise RuntimeError('explicit serial not ready')
        props = self.shell('getprop')
        p = dict(re.findall(r'^\[([^\]]+)\]: \[(.*)\]$', props, re.M))
        abi = p.get('ro.product.cpu.abi')
        model = p.get('ro.product.model', '')
        sdk = int(p.get('ro.build.version.sdk', '0'))
        if sdk < 34 or p.get('sys.boot_completed') != '1':
            raise RuntimeError('Android API 34+ and a completed boot are required')
        emulator = (p.get('ro.kernel.qemu') == '1' or p.get('ro.boot.qemu') == '1' or
                    p.get('ro.hardware') in ('ranchu', 'goldfish') or model.startswith('sdk_gphone') or
                    self.args.serial.startswith('emulator-'))
        if self.args.emulator:
            if not (self.args.serial.startswith('emulator-') and emulator and abi == 'x86_64'):
                raise RuntimeError('rehearsal requires a local x86_64 emulator serial')
        elif emulator or abi != 'arm64-v8a' or not model.startswith(self.args.expect_model):
            raise RuntimeError(f'physical arm64/model guard failed: {model} {abi} emulator={emulator}')
        self.result['device'] = dict(model=model, abi=abi, sdk=sdk, emulator=emulator)
        for setting in ('enabled_accessibility_services', 'accessibility_enabled', 'show_ime_with_hard_keyboard'):
            value = self.shell('settings', 'get', 'secure', setting)
            if setting == 'enabled_accessibility_services' and value not in ('null', '', 'app.murmur/app.murmur.platform.a11y.MurmurAccessibilityService'):
                if self.args.mode != 'launch':
                    raise RuntimeError('another accessibility service is active; use a disposable test device')
            self.settings.append((setting, value))

    def installed_apk(self, package):
        out = self.shell('pm', 'path', package, allow=True)
        paths = [l.removeprefix('package:') for l in out.splitlines() if l.startswith('package:')]
        if not paths:
            return None
        if len(paths) != 1:
            raise RuntimeError(f'{package}: split APK is not the staged candidate')
        return self.shell('sha256sum', paths[0], budget=20).split()[0]

    def charge(self, entry):
        self.spent += entry['bytes']
        if self.spent > self.args.max_bytes:
            raise RuntimeError('payload cap exceeded before transfer')
        print(f"transfer {entry['bytes']:,} B: {entry['path']}", flush=True)

    def remote_model(self, entry):
        base = f"no_backup/models/{entry['id']}"
        marker = self.as_app(f"if [ -f {base}/installed.json ]; then cat {base}/installed.json; elif [ -e {base} ]; then echo BROKEN; fi")
        if not marker:
            return False
        try:
            m = json.loads(marker)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"invalid installed model marker: {entry['id']}") from e
        expected = (entry['id'], entry['catalog_version'], entry['files'], entry['raw_bytes'])
        if tuple(m.get(k) for k in ('artifactId', 'catalogVersion', 'files', 'bytesOnDisk')) != expected:
            raise RuntimeError(f"installed model metadata mismatch: {entry['id']}")
        checks = self.as_app('cd ' + base + ' && sha256sum ' + ' '.join(entry['files']), budget=20)
        observed = {name: sha for sha, name in (line.split() for line in checks.splitlines())}
        if observed != entry['files']:
            raise RuntimeError(f"installed model bytes mismatch: {entry['id']}")
        return True

    def launch(self):
        log = self.shell('am', 'start', '-W', '-n', PKG + '/.MainActivity', budget=18)
        (self.out / 'launch.txt').write_text(log + '\n')
        if 'Status: ok' not in log:
            raise RuntimeError(f'MainActivity did not launch: {log[-500:]}')
        activities = self.shell('dumpsys', 'activity', 'activities')
        if not any(PKG + '/' in line and 'MainActivity' in line for line in activities.splitlines()
                   if 'ResumedActivity' in line):
            raise RuntimeError('MainActivity was not the resumed activity (unlock device/check dialogs)')
        self.result['launched'] = True
        print('MainActivity resumed', flush=True)

    def execute(self, apks, models):
        self.guard()
        install = []
        for package, entry in apks:
            got = self.installed_apk(package)
            if got is None:
                install.append((package, entry))
            elif got != entry['sha256']:
                raise RuntimeError(f'{package}: installed APK differs; will not overwrite a different build')
        have_app = not any(package == PKG for package, _ in install)
        models_to_send = [entry for entry in models if not have_app or not self.remote_model(entry)]
        needed = sum(e['bytes'] for _, e in install) + sum(e['bytes'] for e in models_to_send)
        self.result['planned_payload'] = needed
        self.result['planned_files'] = [e['path'] for _, e in install] + [e['path'] for e in models_to_send]
        print(f'planned upload {needed:,} / {self.args.max_bytes:,} B', flush=True)
        if needed > self.args.max_bytes:
            raise RuntimeError('payload cap would be exceeded; no transfer attempted (choose launch/insertion or reuse exact files)')
        if self.args.plan:
            return
        if not self.args.disposable and self.args.mode != 'launch':
            raise RuntimeError('pass --disposable for instrumentation on a dedicated test device')
        def install_apk(package, entry):
            self.charge(entry)
            output = self.call('install', '--streaming', '-r', '-t', str(self.args.directory / entry['path']), budget=75)
            if 'Success' not in output or self.installed_apk(package) != entry['sha256']:
                raise RuntimeError(f'install verification failed: {package}')

        if not have_app:
            install_apk(*install[0])
        if self.args.mode == 'launch':
            self.launch()
            return
        self.changed = True
        self.shell('am', 'force-stop', PKG)
        self.as_app('mkdir -p files; rm -f files/debug-fake-container files/debug-fake-engines files/debug-fake-engines-profile files/debug-capture.wav')
        self.shell('pm', 'grant', PKG, 'android.permission.RECORD_AUDIO')
        self.shell('pm', 'grant', PKG, 'android.permission.POST_NOTIFICATIONS')
        self.shell('settings', 'put', 'secure', 'show_ime_with_hard_keyboard', '1')
        self.launch()
        self.shell('am', 'force-stop', PKG)
        for package, entry in install:
            if package != PKG:
                install_apk(package, entry)
        for entry in models_to_send:
            base = f"no_backup/models/{entry['id']}"
            temp = base + '.stream-new'
            self.pending = temp
            script = f'mkdir -p no_backup/models && rm -rf {temp} && mkdir {temp} && tar -xzf - -C {temp} && test -f {temp}/installed.json && mv {temp} {base}'
            self.charge(entry)
            with (self.args.directory / entry['path']).open('rb') as payload:
                self.as_app(script, source=payload, budget=110)
            self.pending = None
            if not self.remote_model(entry):
                raise RuntimeError(f"model transfer failed: {entry['id']}")
        if self.args.mode == 'insertion':
            self.as_app('touch files/debug-fake-engines')
        started = self.shell('date', '+%m-%d\ %H:%M:%S.000'.replace('\\ ', ' '))
        try:
            report = self.shell('am', 'instrument', '-w', '-e', 'class', TEST_CLASS,
                                '-e', 'stream.mode', self.args.mode, '-e', 'stream.emulator',
                                str(int(self.args.emulator)), RUNNER, budget=165)
            (self.out / 'instrument.txt').write_text(report + '\n')
            if not re.search(r'^OK \(1 test\)$', report, re.M) or 'AssumptionViolated' in report:
                raise RuntimeError('instrumentation did not pass; see instrument.txt')
            if 'report stream.pass=' + self.args.mode not in self.call('logcat', '-d', '-T', started, '-e', 'report stream\\.pass=', '-s', 'E2E:I', budget=4):
                raise RuntimeError('missing positive instrumentation completion marker')
        finally:
            try:
                output = self.call('logcat', '-d', '-T', started, '-v', 'brief', '-e', 'report stream\\.', '-s', 'E2E:I', budget=4)
                (self.out / 'evidence.txt').write_text(output[-16384:] + '\n')
            except (RuntimeError, TimeoutError, subprocess.TimeoutExpired):
                pass

    def finish(self):
        if not self.changed:
            return
        jobs = [lambda: self.shell('am', 'force-stop', PKG, budget=2, cleanup=True),
                lambda: self.as_app('rm -f files/debug-fake-engines files/debug-capture.wav' +
                                    (f'; rm -rf {self.pending}' if self.pending else ''), budget=2, cleanup=True)]
        for key, old in self.settings:
            cmd = ('delete', 'secure', key) if old in ('null', '') else ('put', 'secure', key, old)
            jobs.append(lambda cmd=cmd: self.shell('settings', *cmd, budget=2, cleanup=True))
        errors = []
        for job in jobs:
            try:
                job()
            except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as e:
                errors.append(str(e))
        if errors:
            self.result['cleanup_errors'] = errors


def main():
    parser = argparse.ArgumentParser(description='Offline Murmur WAV-to-editor physical arm64 smoke; never connects or rents.')
    os.umask(0o077)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--serial')
    parser.add_argument('--expect-model', help='required physical ro.product.model prefix')
    parser.add_argument('--mode', choices=CAPS, default='ocr')
    parser.add_argument('--max-bytes', type=int, help='upload payload limit (APK + compressed model archives; no retries)')
    parser.add_argument('--seconds', type=int, default=240, help='hard host-side adb deadline, maximum 360')
    parser.add_argument('--preflight', action='store_true', help='validate local files and print worst-case bytes; no ADB')
    parser.add_argument('--plan', action='store_true', help='read-only plan for one already attached serial')
    parser.add_argument('--disposable', action='store_true', help='allow changing test settings on a dedicated target')
    parser.add_argument('--emulator', action='store_true', help='explicit local x86_64 rehearsal, never physical evidence')
    args = parser.parse_args()
    if args.max_bytes is None:
        args.max_bytes = CAPS[args.mode]
    if not (0 <= args.max_bytes <= CAPS[args.mode] and 20 <= args.seconds <= 360):
        parser.error(f'cap must be 0..{CAPS[args.mode]} bytes and 20..360 seconds')
    if not args.preflight and not re.fullmatch(r'[A-Za-z0-9_.:\[\]-]+', args.serial or ''):
        parser.error('one explicit attached --serial is required')
    if not (args.preflight or args.emulator) and not re.fullmatch(r'[A-Za-z0-9_-]+', args.expect_model or ''):
        parser.error('specify the expected physical model prefix')
    args.directory = args.directory.resolve()
    apks, models = inputs(args)
    if args.preflight:
        total = sum(e['bytes'] for _, e in apks) + sum(e['bytes'] for e in models)
        for e in [e for _, e in apks] + models:
            print(f"{e['bytes']:>12,} {e['path']}")
        print(f'{args.mode}: cold upload {total:,} B; cap {args.max_bytes:,} B; no adb used')
        return int(total > args.max_bytes)
    out = args.directory / 'runs' / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + args.mode)
    out.mkdir(parents=True)
    print(f'report: {out}', flush=True)
    run = DeviceRun(args, out)
    start = time.monotonic()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        run.execute(apks, models)
        run.result['success'] = True
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as e:
        run.result['error'] = str(e) or type(e).__name__
        print(run.result['error'], file=sys.stderr)
    finally:
        run.finish()
        if run.result.get('cleanup_errors'):
            run.result['success'] = False
        run.result.update(upload_payload_upper_bound=run.spent, elapsed_s=round(time.monotonic() - start, 2), plan_only=args.plan)
        (out / 'result.json').write_text(json.dumps(run.result, indent=2) + '\n')
    return not run.result['success']


if __name__ == '__main__':
    sys.exit(main())
