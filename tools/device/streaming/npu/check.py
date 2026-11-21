import argparse
import array
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

CASES = {
    'det': ([1, 3, 960, 416], [1, 1, 960, 416], 0.15, 0.001, 0.999),
    'rec': ([1, 3, 48, 320], [1, 40, 852], 0.08, 0.001, 0.975),
}
EP = 'QNNExecutionProvider'


def tensor(path, shape):
    raw = path.read_bytes()
    if len(raw) != 4 * math.prod(shape):
        raise ValueError(f'{path}: wrong tensor byte count {len(raw)} for {shape}')
    values = array.array('f', raw)
    if sys.byteorder != 'little':
        values.byteswap()
    if not all(map(math.isfinite, values)):
        raise ValueError(f'{path}: nonfinite tensor')
    return values


def difference(a, b):
    delta = [abs(x-y) for x, y in zip(a, b)]
    return {'max_abs': max(delta), 'mean_abs': math.fsum(delta) / len(delta)}


def top1(seq, classes):
    return [max(range(classes), key=row.__getitem__) for i in range(0, len(seq), classes)
            for row in (seq[i:i+classes],)]


def agreement(case, shape, a, b):
    if case == 'det':
        return sum((x > 0.2) == (y > 0.2) for x, y in zip(a, b)) / len(a)
    pa, pb = top1(a, shape[-1]), top1(b, shape[-1])
    return sum(x == y for x, y in zip(pa, pb)) / len(pa)


def metadata(path, mode, shape):
    data, = [json.loads(line) for line in path.read_text(errors='replace').splitlines() if line.startswith('{"mode":')]
    if (data['mode'] != mode or data['ort'] != '1.32.0' or data['strict'] is not (mode != 'cpu')
            or data['out_shape'] != shape or data['repeats'] != (1 if mode == 'proof' else 7) or data['threads'] != 4
            or any(not math.isfinite(data[k]) or data[k] <= 0 for k in ('load_ms', 'first_ms', 'warm_median_ms', 'warm_min_ms'))):
        raise ValueError(f'{path}: bad runtime metadata: {data}')
    return data


def proof_events(path):
    events = json.loads(path.read_text())
    kernels = [e['args'].get('provider') for e in events if e.get('cat') == 'Node' and
               e.get('name', '').endswith('_kernel_time')]
    backed = [e for e in events if e.get('args', {}).get('qnn_operation') == 'execute' and
              e['args'].get('qnn_timing_source') == 'BACKEND' and e.get('dur', 0) > 0]
    return {'kernel_providers': sorted(set(str(x) for x in kernels)), 'kernel_events': len(kernels),
            'backend_execute_events': len(backed), 'backend_event_names': sorted({e['name'] for e in backed}),
            'pass': len(kernels) >= 2 and set(kernels) == {EP}}


def accelerator_events(path):
    count = 0
    with path.open(newline='', encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if (len(row) >= 4 and row[2].strip().isdecimal() and int(row[2]) > 0
                    and any('accelerator (execute) time' in cell.strip().lower() for cell in row[3:])):
                count += 1
    return count


def evaluate(root, case, work):
    _, shape, max_abs, mean_abs, min_agree = CASES[case]
    meta = {mode: metadata(work / f'{mode}.log', mode, shape) for mode in ('cpu', 'qnn', 'proof')}
    ref = tensor(root / 'reference' / f'{case}.f32', shape)
    cpu, qnn, proof = (tensor(work / (mode + '.f32'), shape) for mode in meta)
    diff = difference(cpu, qnn)
    agree = agreement(case, shape, cpu, qnn)
    cpu_ref = difference(cpu, ref)
    proof_diff = difference(qnn, proof)
    evidence = proof_events(work / 'proof.json')
    evidence['qnn_csv_accelerator_execute_events'] = accelerator_events(work / 'htp.csv')
    failed = (work / 'broken.log').read_text(errors='replace')
    negative = ((work / 'broken.rc').read_text().strip() == '10' and not (work / 'broken.f32').exists()
                and not re.search(r'^\{"mode":', failed, re.M)
                and bool(re.search(r'^MURMUR_PROBE_FAILURE stage=(options|session):', failed, re.M)))
    gates = {
        'cpu_matches_host_reference': cpu_ref['max_abs'] <= 0.005 and cpu_ref['mean_abs'] <= 0.0001,
        'same_device_cpu_qnn_parity': diff['max_abs'] <= max_abs and diff['mean_abs'] <= mean_abs and agree >= min_agree,
        'profile_matches_timed_output': proof_diff['max_abs'] <= 0.001,
        'qnn_only_profile': evidence['pass'],
        'htp_execution_reported': evidence['qnn_csv_accelerator_execute_events'] >= 2,
        'missing_htp_backend_failed_closed': negative,
    }
    return {'case': case, 'pass': all(gates.values()), 'gates': gates, 'timings': meta,
            'warm_cpu_over_qnn': meta['cpu']['warm_median_ms'] / meta['qnn']['warm_median_ms'],
            'cpu_vs_qnn': dict(diff, agreement=agree, metric='DB map > 0.2' if case == 'det' else 'CTC frame top-1'),
            'thresholds': {'max_abs': max_abs, 'mean_abs': mean_abs, 'agreement': min_agree},
            'cpu_vs_host_reference': cpu_ref, 'timed_qnn_vs_profiled_qnn': proof_diff, 'profile': evidence}


def preflight(root, install):
    entries = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = root / name
        if (not re.fullmatch(r'(payload|reference|app)/[A-Za-z0-9_./+-]+', name)
                or '..' in Path(name).parts or name in entries or path.resolve() != path
                or not re.fullmatch(r'[0-9a-f]{64}', digest)):
            raise ValueError('invalid manifest entry')
        if name.startswith('app/') and not install:
            continue
        with path.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hash_stream(f)
        if actual != digest:
            raise ValueError(f'checksum mismatch: {path}')
        entries[name] = path.stat().st_size
    payload = {str(p.relative_to(root)) for p in (root / 'payload').rglob('*') if p.is_file()}
    if payload != {name for name in entries if name.startswith('payload/')}:
        raise ValueError('payload contains a missing or unsealed file')
    required = {'payload/bin/probe', 'payload/lib/libonnxruntime.so',
                'payload/lib/libonnxruntime_providers_qnn.so', 'payload/lib/libQnnHtp.so',
                'payload/det/det_h960_w416.onnx', 'payload/det/det_h960_w416_qnn.bin',
                'payload/rec/rec_w320.onnx', 'payload/rec/rec_w320_qnn.bin'}
    required.update(f'payload/{case}/{name}' for case in CASES for name in ('cpu.onnx', 'input.f32'))
    required.update(f'reference/{case}.f32' for case in CASES)
    if install:
        required.add('app/murmur-npu-debug.apk')
    if missing := required - entries.keys():
        raise ValueError(f'missing manifest entries: {sorted(missing)}')
    for case, (ins, outs, _, _, _) in CASES.items():
        tensor(root / 'payload' / case / 'input.f32', ins)
        tensor(root / 'reference' / f'{case}.f32', outs)
    print(json.dumps({'preflight': 'PASS', 'verified_files': len(entries),
                      'payload_bytes': sum(entries[p] for p in payload), 'apk_verified': install}))


def hash_stream(f):
    digest = hashlib.sha256()
    for chunk in iter(lambda: f.read(1048576), b''):
        digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('preflight')
    p.add_argument('--app', action='store_true')
    e = sub.add_parser('evaluate')
    e.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.command == 'preflight':
        preflight(args.bundle.resolve(), args.app)
        return 0
    rows = []
    for case in CASES:
        try:
            rows.append(evaluate(args.bundle.resolve(), case, args.directory / case))
        except (ValueError, TypeError, OSError, KeyError) as exc:
            rows.append({'case': case, 'pass': False, 'error': str(exc)})
    print(json.dumps({'pass': all(r['pass'] for r in rows), 'cases': rows}, indent=2))
    return 0 if all(r['pass'] for r in rows) else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))
