"""Read-only dataset audit: scores, exports, duplicates and bootstrap sensitivity.

Does not execute candidate code or query any model. Writes audit artifacts only.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from melchior.crucible.decisions import comparison, nli_rows
from melchior.crucible.metrics import paired_interval, score_predictions
from melchior.crucible.prompts import select_operator_by_seed


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def bootstrap_deltas(metric, y, a, b, seed, samples=5000):
    """Independent, vectorized implementation; same stratified sampling design."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in np.unique(y)] if metric != 'r2' else [np.arange(len(y))]
    chunks = []
    for start in range(0, samples, 100):
        size = min(100, samples-start)
        ix = np.concatenate([rng.choice(g, (size, len(g)), replace=True) for g in groups], axis=1)
        yy = y[ix]
        def scores(p):
            pp = p[ix]
            if metric == 'accuracy':
                return (pp == yy).mean(axis=1)
            if metric == 'f1':
                tp = ((pp == 1) & (yy == 1)).sum(axis=1)
                denom = (pp == 1).sum(axis=1) + (yy == 1).sum(axis=1)
                return np.divide(2*tp, denom, out=np.zeros(size, dtype=float), where=denom != 0)
            if metric == 'r2':
                return 1-((yy-pp)**2).sum(axis=1)/((yy-yy.mean(axis=1, keepdims=True))**2).sum(axis=1)
            if metric == 'roc_auc':
                ranks = rankdata(pp, axis=1, method='average')
                positive = yy == 1
                npos = positive.sum(axis=1)
                nneg = yy.shape[1]-npos
                return ((ranks*positive).sum(axis=1)-npos*(npos+1)/2)/(npos*nneg)
            raise ValueError(metric)
        chunks.append(scores(b)-scores(a))
    return np.concatenate(chunks)


def code_info(code):
    tree = ast.parse(code)
    imports = sorted({n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} |
                     {x.name for n in ast.walk(tree) if isinstance(n, ast.Import) for x in n.names})
    constructors = sorted({n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and
                           isinstance(n.func, ast.Name) and n.func.id[:1].isupper()})
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    risky = sorted(names & {'open', 'eval', 'exec', '__import__', 'y_test', 'y_val', 'y_validation'})
    risky += [s for s in imports if s.split('.')[0] not in {'numpy','scipy','sklearn','pandas','math','warnings','typing'}]
    if any(s.startswith('sklearn.datasets') for s in imports):
        risky.append('sklearn.datasets')
    return {'ast_hash': hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest(),
            'constructors': constructors, 'imports': imports, 'review_names_or_imports': risky,
            'lines': len(code.splitlines())}


def audit(root, output):
    load = lambda name: [json.loads(l) for l in (root/name).read_text().splitlines() if l.strip()]
    records = load('evaluations.jsonl')
    nli, dpo = load('openjev_ml_nli.jsonl'), load('melchior_dpo_pairs.jsonl')
    manifest = json.loads((root/'manifest.json').read_text())
    minimum = manifest['min_delta']
    issues = []
    accepted = []
    expected_nli, expected_dpo = [], []
    operator_counts, family_counts = defaultdict(Counter), defaultdict(Counter)
    all_scored = 0
    for r in records:
        op = select_operator_by_seed(r['seed'])
        family = r['profile']['dataset_id'] if not r['profile']['dataset_id'].startswith('synthetic:') else 'synthetic_variant_'+str(r['seed'] % 6)
        for counts in (operator_counts[op], family_counts[family]):
            counts['attempted'] += 1
            counts['generated'] += 'pair' in r
            counts['generation_failed'] += 'error' in r
        if 'outcome' not in r:
            continue
        winner = r['outcome']['winner']
        for counts in (operator_counts[op], family_counts[family]):
            counts[winner] += 1
        y = np.asarray(r['test_labels'])
        for key in ('res_a', 'res_b'):
            if r[key]['status'] == 'success':
                score = score_predictions(r['profile']['metric'], y, r[key]['predictions'])
                all_scored += 1
                if not np.isclose(score, r[key]['metric'], rtol=0, atol=1e-12):
                    issues.append([r['seed'], key, 'score_mismatch', score, r[key]['metric']])
        if winner not in ('A', 'B'):
            if r['outcome']['nli_records'] or r['outcome']['dpo_record']:
                issues.append([r['seed'], 'nondecisive_export'])
            continue
        assert r['mode'] != 'mock' and r['profile']['split'] == 'train'
        assert all(r[k]['status'] == 'success' for k in ('res_a','res_b'))
        a, b = np.asarray(r['res_a']['predictions']), np.asarray(r['res_b']['predictions'])
        metric = r['profile']['metric']
        interval = paired_interval(metric, y, a, b, seed=r['seed'])
        predicted = 'B' if interval[0] > minimum else 'A' if interval[1] < -minimum else 'UNCERTAIN'
        if predicted != winner or not np.allclose(interval, r['outcome']['interval'], rtol=0, atol=1e-12):
            issues.append([r['seed'], 'interval_or_winner_mismatch', interval, winner])
        payload = comparison(r['profile'], r['pair'])
        expected_nli.extend(nli_rows(payload, winner))
        chosen, rejected = ('a','b') if winner == 'A' else ('b','a')
        expected_dpo.append({'prompt': payload['state'], 'chosen': r['pair']['code_'+chosen],
                             'rejected': r['pair']['code_'+rejected],
                             'margin': abs(r['res_b']['metric']-r['res_a']['metric']), 'operator':op})
        if r['outcome']['nli_records'] != nli_rows(payload, winner) or r['outcome']['dpo_record'] != expected_dpo[-1]:
            issues.append([r['seed'], 'embedded_export_mismatch'])
        deltas = bootstrap_deltas(metric, y, a, b, r['seed']+1000000)
        ci = np.quantile(deltas, [.025,.975]).tolist()
        winner5000 = 'B' if ci[0] > minimum else 'A' if ci[1] < -minimum else 'UNCERTAIN'
        codes = {k: code_info(r['pair']['code_'+k]) for k in ('a','b')}
        accepted.append({'seed':r['seed'], 'operator':op, 'family':family, 'dataset_id':r['profile']['dataset_id'],
                         'data_hash':r['profile']['data_hash'], 'metric':metric, 'winner':winner,
                         'score_a':r['res_a']['metric'], 'score_b':r['res_b']['metric'],
                         'margin':abs(r['res_b']['metric']-r['res_a']['metric']),
                         'interval_200':interval, 'interval_5000':ci, 'winner_5000':winner5000,
                         'n_test':len(y), 'class_counts':dict(Counter(map(str,y))) if metric!='r2' else None,
                         'codes':codes, 'generation_swapped':r['generation_swapped'],
                         'original_winner': ('B' if winner=='A' else 'A') if r['generation_swapped'] else winner})
        if len(accepted) % 25 == 0:
            print(f'Audited {len(accepted)} accepted pairs', flush=True)
    if expected_nli != nli:
        issues.append(['NLI export mismatch'])
    if expected_dpo != dpo:
        issues.append(['DPO export mismatch'])
    labels_by_proposition = defaultdict(set)
    for row in nli:
        labels_by_proposition[(row['premise'],row['hypothesis'])].add(row['label'])
    def duplicates(values):
        c=Counter(values)
        return {'unique':len(c),'extra_copies':sum(v-1 for v in c.values()),'max_multiplicity':max(c.values(),default=0)}
    q = lambda values: np.quantile(values,[0,.1,.5,.9,1]).tolist()
    output.mkdir(parents=True,exist_ok=True)
    (output/'pair_details.json').write_text(json.dumps(accepted,indent=2))
    report = {
        'source_manifest':manifest, 'record_count':len(records), 'accepted_count':len(accepted),
        'recomputed_successful_scores':all_scored, 'integrity_issues':issues,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.json*')},
        'operator_counts':dict(operator_counts), 'family_counts':dict(family_counts),
        'accepted_metrics':dict(Counter(a['metric'] for a in accepted)),
        'accepted_winners':dict(Counter(a['winner'] for a in accepted)),
        'original_winners':dict(Counter(a['original_winner'] for a in accepted)),
        'accepted_families':dict(Counter(a['family'] for a in accepted)),
        'independent_dataset_groups':len({a['dataset_id'] for a in accepted}),
        'bootstrap5000_changes':[{'seed':a['seed'],'old':a['winner'],'new':a['winner_5000'],'ci200':a['interval_200'],'ci5000':a['interval_5000']} for a in accepted if a['winner']!=a['winner_5000']],
        'duplicates':{'seeds':duplicates(r['seed'] for r in records), 'accepted_data_hashes':duplicates(a['data_hash'] for a in accepted),
                      'nli_rows':duplicates(canonical(r) for r in nli), 'nli_premises':duplicates(r['premise'] for r in nli),
                      'dpo_rows':duplicates(canonical(r) for r in dpo),
                      'candidate_asts':duplicates(a['codes'][k]['ast_hash'] for a in accepted for k in ('a','b')),
                      'unordered_code_pairs':duplicates(tuple(sorted(a['codes'][k]['ast_hash'] for k in ('a','b'))) for a in accepted)},
        'conflicting_propositions':sum(len(labels)>1 for labels in labels_by_proposition.values()),
        'nli_labels':dict(Counter(r['label'] for r in nli)),
        'nli_exact_schema':all(set(r)=={'premise','hypothesis','label','source','image'} for r in nli),
        'margin_quantiles_by_metric':{m:q([a['margin'] for a in accepted if a['metric']==m]) for m in sorted({a['metric'] for a in accepted})},
        'negative_r2_pairs':[a['seed'] for a in accepted if a['metric']=='r2' and min(a['score_a'],a['score_b'])<0],
        'test_size_quantiles':q([a['n_test'] for a in accepted]),
        'code_review_flags':[{'seed':a['seed'],'side':k,'flags':a['codes'][k]['review_names_or_imports']} for a in accepted for k in ('a','b') if a['codes'][k]['review_names_or_imports']],
        'constructors':dict(Counter(c for a in accepted for k in ('a','b') for c in a['codes'][k]['constructors'])),
    }
    (output/'audit_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('constructors','source_manifest','source_sha256')},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('data',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    audit(args.data,args.output)
