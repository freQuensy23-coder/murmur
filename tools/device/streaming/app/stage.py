#!/usr/bin/env python3
import argparse
import gzip
import os
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CAT = ROOT / 'core/catalog/src/main/resources/catalog/catalog.json'
IDS = (
    'gigaam-v3-e2e-rnnt@onnx-int8', 'silero-vad-v5@onnx',
    'pp-ocrv6-tiny-det@onnx', 'pp-ocrv5-cyrillic-mobile-rec@onnx', 'qwen3.5-0.8b@q4_0',
)


def item(path, directory):
    with path.open('rb') as f:
        return dict(path=str(path.relative_to(directory)), bytes=path.stat().st_size,
                    sha256=hashlib.file_digest(f, 'sha256').hexdigest())


def main():
    parser = argparse.ArgumentParser(description="Stage local APKs and catalog models; no device access.")
    parser.add_argument("directory", type=Path)
    for key in ("app", "test", "target"):
        parser.add_argument(f"--{key}-apk", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=ROOT / ".cache/models")
    parser.add_argument("--tier", choices=("launch", "insertion", "ocr", "full"), default="ocr")
    args = parser.parse_args()
    os.umask(0o077)
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cat = json.loads(CAT.read_text())
    by_id = {a['id']: a for m in cat['models'] for a in m['artifacts']}
    apkdir = directory / 'apks'
    apkdir.mkdir(exist_ok=True)
    for name in ('app', 'test', 'target'):
        shutil.copyfile(getattr(args, name + '_apk'), apkdir / f'{name}.apk')
    modeldir = directory / 'models'
    modeldir.mkdir(exist_ok=True)
    models = []
    for aid in IDS[:{'launch': 0, 'insertion': 0, 'ocr': 4, 'full': 5}[args.tier]]:
        files = by_id[aid]['files']
        source = args.models / aid
        archive = modeldir / f'{aid}.tgz'
        signatures = {f['name']: f['sha256'] for f in files}
        raw = sum(f['bytes'] for f in files)
        marker = dict(artifactId=aid, catalogVersion=cat['catalogVersion'], files=signatures,
                      bytesOnDisk=raw, installedAtMs=0, prepared=True)
        for f in files:
            p = source / f['name']
            with p.open('rb') as src:
                if p.stat().st_size != f['bytes'] or hashlib.file_digest(src, 'sha256').hexdigest() != f['sha256']:
                    raise ValueError(f'catalog hash/size mismatch: {p}')
        with archive.open('wb') as out, gzip.GzipFile(filename='', fileobj=out, mode='wb', compresslevel=1, mtime=0) as gz, tarfile.open(fileobj=gz, mode='w|') as tar:
            for f in files:
                info = tarfile.TarInfo(f['name'])
                info.size = f['bytes']
                with (source / f['name']).open('rb') as src:
                    tar.addfile(info, src)
            data = json.dumps(marker, sort_keys=True).encode()
            info = tarfile.TarInfo('installed.json')
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        models.append(dict(item(archive, directory), id=aid, raw_bytes=raw, files=signatures, catalog_version=cat['catalogVersion']))
    apks = {p: item(apkdir / f'{p}.apk', directory) for p in ('app', 'test', 'target')}
    (directory / 'manifest.json').write_text(json.dumps(dict(apks=apks, models=models), indent=2) + '\n')
    for p, data in apks.items():
        print(p, data['bytes'])
    for data in models:
        print(data['id'], data['raw_bytes'], data['bytes'])


if __name__ == '__main__':
    main()
