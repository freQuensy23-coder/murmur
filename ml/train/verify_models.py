import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path
from ml.train.config import pinned_revisions


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'Murmur-verification/1'})
    with urllib.request.urlopen(request, timeout=45) as r:
        return r.read(), r.headers

def inspect(item):
    repo, sha, directory = item
    payload, _ = get(f'https://huggingface.co/api/models/{repo}/revision/{sha}?blobs=true')
    model = json.loads(payload)
    if model['sha'] != sha or model['private'] or model['gated']:
        raise ValueError(f'{repo}: revision must be public, ungated and immutable')
    matches = {}
    for filename in ('config.json', 'chat_template.jinja', 'tokenizer_config.json', 'tokenizer.json'):
        b, h = get(f'https://huggingface.co/{repo}/resolve/{sha}/{filename}')
        local = directory / filename
        if b != local.read_bytes():
            raise ValueError(f'{repo}: {filename} does not match the pinned revision')
        matches[filename] = {'sha256': hashlib.sha256(b).hexdigest(), 'bytes': len(b), 'x-repo-commit': h.get('x-repo-commit')}
    weights = {f['rfilename']: f['lfs'] for f in model['siblings'] if f['rfilename'].endswith('.safetensors')}
    checked = []
    for filename, meta in weights.items():
        if (p := directory / filename).is_file():
            with p.open('rb') as f:
                if p.stat().st_size != meta['size'] or hashlib.file_digest(f, 'sha256').hexdigest() != meta['sha256']:
                    raise ValueError(f'{repo}: weight shard {filename} does not match the pinned revision')
            checked.append(filename)
    return repo, {'revision':sha, 'matches':matches, 'weights':weights, 'local_weights_verified':checked}

def main():
    parser = argparse.ArgumentParser(description='Compare local model files against public pinned Hub revisions.')
    parser.add_argument('--model', nargs=2, metavar=('REPO', 'DIRECTORY'), action='append', required=True)
    args = parser.parse_args()
    pins = pinned_revisions()
    if any(repo not in pins for repo, _ in args.model) or len(dict(args.model)) != len(args.model):
        parser.error('give unique repository names pinned by the training config')
    with concurrent.futures.ThreadPoolExecutor() as pool:
        jobs = ((repo, pins[repo], Path(directory)) for repo, directory in args.model)
        print(json.dumps(dict(pool.map(inspect, jobs)), indent=2))


if __name__ == '__main__':
    main()
