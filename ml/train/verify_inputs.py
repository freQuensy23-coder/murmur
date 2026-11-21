import argparse
import hashlib
import itertools
import json
import unicodedata
from collections import Counter
from pathlib import Path

from ml.data.recheck import rule_errors
from ml.eval import contract

def canonical(v):
    return ' '.join(unicodedata.normalize('NFKC', v).casefold().split())

def fields(rows):
    return {k: set(f(r) for r in rows) - {None, ''} for k, f in queries.items()}

queries = {
    'id': lambda r: r['id'], 'group': lambda r: r['meta'].get('group'),
    'scenario': lambda r: r['meta'].get('scenario_id') or r['meta'].get('scenario'),
    'raw_pair': lambda r: (r['transcript'], r['target']),
    'normalized_pair': lambda r: (canonical(r['transcript']), canonical(r['target'])),
    'render_input': lambda r: contract.render_prompt(r, 'finetuned.v2', 'lfm3'),
    'transcript': lambda r: canonical(r['transcript']), 'target': lambda r: canonical(r['target']),
    'spoken': lambda r: canonical(r['meta'].get('spoken') or ''),
}

def audit(root, source_dir, holdouts):
    splits = ('train', 'dev', 'test')
    recorded = json.loads((root / 'stats.json').read_text())
    result, views = {'disjoint': True}, {}
    for split in splits:
        file = root/f'{split}.jsonl'
        rows = contract.load(file)
        byid = {r['id']: r for r in rows}
        source = {r['id']: r for r in contract.load(source_dir/file.name)}
        stats = {'n':len(rows), **{k: dict(Counter(r[k] for r in rows)) for k in ('lang', 'source', 'category')}, 'empty_target': sum(not r['target'].strip() for r in rows)}
        errors = [err for r in rows for err in rule_errors(r)]
        if not all(r['split'] == split and r['ctx']['v'] == 2 and r['meta'].get('group') for r in rows):
            raise ValueError(f'{split}: invalid split, context version, or group')
        if stats != recorded[split] or byid != source or len(byid) != len(rows) or errors:
            raise ValueError(f'{split}: statistics, sources, unique IDs or rules did not match')
        if not rows:
            raise ValueError(f'{split}: empty split')
        views[split] = fields(rows)
        result[split] = {**stats, 'sha256': hashlib.sha256(file.read_bytes()).hexdigest(), 'bytes': file.stat().st_size,
                         'unique_rendered_inputs':len(views[split]['render_input']), 'context_rule_failures': len(errors), 'distinct_generator_labels': len({(r['meta'].get('teacher'), r['meta'].get('gen')) for r in rows})}
    for a,b in itertools.combinations(splits,2):
        result[f'{a}/{b}'] = {k:len(views[a][k]&views[b][k]) for k in queries}
        if any(result[f'{a}/{b}'][k] for k in ('id','group','scenario','raw_pair','normalized_pair','render_input')):
            result['disjoint'] = False
    for i, p in enumerate(holdouts, 1):
        official = fields(contract.load(p))
        result[f'holdout_{i}'] = {k:len(views['train'][k]&official[k]) for k in queries}
    return result

def main():
    parser = argparse.ArgumentParser(description='Read-only input audit; outputs aggregate results only.')
    parser.add_argument('assembled', type=Path)
    parser.add_argument('source_split_dir', type=Path)
    parser.add_argument('--holdout', type=Path, action='append', default=[])
    args = parser.parse_args()
    result = audit(args.assembled, args.source_split_dir, args.holdout)
    print(json.dumps(result, indent=2))
    return 0 if result['disjoint'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
