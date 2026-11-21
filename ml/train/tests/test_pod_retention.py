import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml.train import upload_hf

REPO = Path(__file__).resolve().parents[3]


def write(path, content, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    path.chmod(mode)
    return path


def gguf(run):
    path = write(run / 'gguf/model-Q4_0.gguf', 'q4-weights')
    manifest = {'name': 'model', 'source': str(run / 'merged'), 'files': [
        {'quant': 'Q4_0', 'path': path.name, 'bytes': path.stat().st_size,
         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]}
    write(path.parent / 'manifest.json', json.dumps(manifest))
    return path


@pytest.fixture
def retained(tmp_path):
    if not shutil.which('rsync'):
        pytest.skip('rsync unavailable')
    remote = tmp_path / 'volume'
    run = remote / 'runs/anchor'
    write(remote / 'runs/finalists.txt', 'Org/model anchor\n')
    gguf(run)
    gguf(remote / 'runs/nonfinalist')
    write(remote / 'runs/nonfinalist/final/model.safetensors', 'full-finetune')
    write(run / 'final/adapter/adapter_model.safetensors', 'adapter')
    write(run / 'final/adapter/adapter_config.json', '{}')
    write(run / 'final/tokenizer.json', '{}')
    write(run / 'final/chat_template.jinja', 'jinja')
    write(run / 'checkpoint-100/trainer_state.json', '{}')
    write(remote / 'data/train.jsonl', 'should not sync')
    write(remote / 'data/test.jsonl', 'eval data')
    write(remote / 'eval/report.json', '{}')
    write(remote / 'logs/pod.log', 'log')
    bin_dir = tmp_path / 'bin'
    write(bin_dir / 'ssh', '''#!/bin/sh
while [ "$1" = -p ] || [ "$1" = -o ] || [ "$1" = -i ]; do shift 2; done
[ "$1" = fake-pod ] || exit 99
shift
exec sh -c "$*"
''', 0o755)
    dst = tmp_path / 'result'
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'POD': 'fake-pod',
           'POD_WORKSPACE': str(remote), 'DST': str(dst), 'WITH_GGUF': '1', 'WITH_ADAPTERS': '1'}
    return remote, dst, env


def sync(env):
    return subprocess.run(['bash', 'ml/pod/sync_results.sh'], cwd=REPO, env=env, capture_output=True, text=True)


def test_sync_private_and_verifies_selected_artifacts(retained):
    _, dst, env = retained
    r = sync(env)
    assert r.returncode == 0, r.stderr
    assert (dst / 'runs/anchor/final/adapter/adapter_model.safetensors').read_text() == 'adapter'
    assert (dst / 'runs/anchor/final/chat_template.jinja').is_file()
    assert (dst / 'runs/anchor/gguf/model-Q4_0.gguf').read_text() == 'q4-weights'
    assert (dst / 'data/test.jsonl').is_file()
    for missing in ('runs/nonfinalist/gguf/model-Q4_0.gguf', 'runs/anchor/checkpoint-100',
                    'data/train.jsonl', 'runs/nonfinalist/final/model.safetensors'):
        assert not (dst / missing).exists()
    assert all(stat.S_IMODE(p.stat().st_mode) == (0o700 if p.is_dir() else 0o600)
               for p in [dst, *dst.rglob('*')])
    r = sync({**env, 'WITH_FULL': '1'})
    assert r.returncode == 0, r.stderr
    assert (dst / 'runs/nonfinalist/final/model.safetensors').read_text() == 'full-finetune'


@pytest.mark.parametrize('bad', ['missing', 'wrong-hash', 'missing-receipt'])
def test_sync_refuses_incomplete_finalists(retained, bad):
    remote, _, env = retained
    path = remote / 'runs/anchor/gguf/model-Q4_0.gguf'
    if bad == 'missing':
        path.unlink()
    elif bad == 'wrong-hash':
        path.write_text('x' * path.stat().st_size)
    else:
        (remote / 'runs/finalists.txt').unlink()
    r = sync(env)
    assert r.returncode != 0
    assert {'missing': 'missing regular', 'wrong-hash': 'does not match manifest',
            'missing-receipt': 'finalists.txt'}[bad] in r.stderr


def test_cards_require_cpu_report(retained):
    _, dst, env = retained
    r = sync(env)
    assert r.returncode == 0, r.stderr
    write(dst / 'runs/anchor/eval/Q4_0-test/report.json', '{}')
    r = subprocess.run(['bash', 'ml/pod/after_pod.sh'], cwd=REPO, capture_output=True, text=True,
                       env={**env, 'MURMUR_POD_RESULTS': str(dst), 'STEPS': 'cards', 'MURMUR_PYTHON': sys.executable})
    assert r.returncode == 2 and 'missing CPU ship-gate report' in r.stderr


def test_budget_refusal_leaves_current_finalist_receipt(retained):
    remote, _, env = retained
    run = remote / 'runs/anchor'
    write(remote / 'runs/finalists.txt', 'Org/model stale\n')
    write(run / 'train_info.json', '{"base_model":"Org/model"}')
    write(run / 'config.resolved.json', '{}')
    write(run / 'eval/Q4_0-dev/report.json', json.dumps({
        'gates': {'scopes': {'overall': []}, 'verdict': 'INCOMPLETE'}, 'summary': {'overall': {'em': 1.0}}}))
    write(remote / 'data/smoke.v2.test.jsonl', '')
    (remote / 'venv-train/bin').mkdir(parents=True)
    (remote / 'venv-train/bin/python').symlink_to(sys.executable)
    r = subprocess.run(['bash', 'ml/pod/run_all.sh'], cwd=REPO, capture_output=True, text=True, env={
        **env, 'STAGES': 'eval', 'MURMUR_WORKSPACE': str(remote), 'MURMUR_REPO': str(REPO),
        'MURMUR_CONTAINER_MARKERS': str(remote / 'ctr'), 'MURMUR_SESSION_START': str(int(time.time()) - 600),
        'MAX_GPU_HOURS': '0.25', 'BUDGET_TAIL_S': '0', 'BUDGET_SESSION_OVERHEAD_S': '400',
    })
    assert r.returncode == 5, r.stderr
    assert (remote / 'runs/finalists.txt').read_text() == 'Org/model anchor\n'
    assert not (run / 'eval/Q4_0-test').exists()
    assert sync(env).returncode == 0


@pytest.mark.parametrize(('cuda', 'rc'), [('13.0', 3), ('13.1', 3), ('12.10', 3), ('N/A', 3), ('13.2', 42), ('13.10', 42)])
def test_bootstrap_driver_gate_before_installs(tmp_path, cuda, rc):
    bin_dir = tmp_path / 'bin'
    write(bin_dir / 'nvidia-smi', f'#!/bin/sh\nprintf "CUDA Version: {cuda}\\n"\n', 0o755)
    write(bin_dir / 'uv', '#!/bin/sh\nexit 42\n', 0o755)
    volume, ctr = tmp_path / 'volume', tmp_path / 'ctr'
    for p in (ctr / 'apt', ctr / 'net', volume / '.bootstrap/uv', volume / '.bootstrap/llama'):
        write(p, '')
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}', 'MURMUR_WORKSPACE': str(volume),
           'MURMUR_CONTAINER_MARKERS': str(ctr)}
    r = subprocess.run(['bash', 'ml/pod/bootstrap.sh'], cwd=REPO, capture_output=True, text=True, env=env)
    assert r.returncode == rc, r.stderr
    assert ('Need native CUDA' in r.stderr) == (rc == 3)


def test_upload_requires_explicit_private_approval(tmp_path, monkeypatch, capsys):
    calls = []

    class PublicRepo:
        def __init__(self, **kwargs):
            calls.append('api')

        def create_repo(self, *args, **kwargs):
            assert kwargs['private'] is True
            calls.append('create')

        def model_info(self, *args, **kwargs):
            return SimpleNamespace(private=False)

        def upload_file(self, **kwargs):
            pytest.fail('upload against a public repo')

    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=PublicRepo))
    monkeypatch.setenv('HF_TOKEN', 'local-test-only')
    gguf(tmp_path)
    args = ['--run', str(tmp_path), '--repo', 'owner/repo']
    assert upload_hf.main([*args, '--dry-run']) == 0
    assert json.loads(capsys.readouterr().out)['private'] is True
    for flags in ([], ['--public']):
        with pytest.raises(SystemExit) as exc:
            upload_hf.main(args + flags)
        assert exc.value.code == 2
    assert not calls
    with pytest.raises(SystemExit, match='destination is not private'):
        upload_hf.main([*args, '--allow-private-upload'])
    assert calls == ['api', 'create']
