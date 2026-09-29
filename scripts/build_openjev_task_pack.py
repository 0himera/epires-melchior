"""Freeze fresh Qwen canary inputs before consulting any judge scores."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np

from melchior.crucible.environments import generate_task
from melchior.crucible.storage import atomic_json
from scripts.build_openjev_independent_test import real_task


SYNTHETIC_SEEDS = tuple(s for s in range(90000, 90072) if s % 6 < 4)
REAL_DATASETS = (
    (54, 'vehicle', 'multiclass_classification', 'accuracy', None),
    (1462, 'banknote-authentication', 'binary_classification', 'roc_auc', None),
    (1461, 'bank-marketing', 'binary_classification', 'roc_auc', None),
)
REAL_SPLITS = (2718, 2719, 2720, 2721)


def build(output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'arrays').mkdir()
    protocol = {
        'schema': 'qwen-trained-canary-protocol-v1', 'split': 'eval',
        'synthetic_seeds': SYNTHETIC_SEEDS, 'real_datasets': REAL_DATASETS,
        'real_split_seeds': REAL_SPLITS, 'candidate_source': 'fresh Qwen generation',
        'selection': 'All attempted tasks; decisive pairs use paired bootstrap, min_delta=0.005',
        'reasoning_effort': 'xhigh', 'enable_thinking': True, 'generation_max_tokens': 16384,
        'concurrency': 32, 'max_hours': 3, 'sandbox_timeout_s': 60,
        'generation_timeout_s': 1200, 'model_selection': 's42_c00 epoch_2 fixed before generation',
        'primary_score': 'symmetric choice, mean relative entailment in both A/B orientations',
        'comparison': 'After collection, rescore the same pairs with the original base model',
        'independence': 'New synthetic realizations and three previously unused OpenML datasets',
        'grouping': 'OpenML splits share a dataset group; synthetic realizations are separate groups',
        'preprocessing': 'Explicit ID removal only; numeric medians and one-hot encoding fit on train',
        'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__).relative_to(Path.cwd()), Path('scripts/build_openjev_independent_test.py'),
            Path('melchior/crucible/environments.py'))},
    }
    atomic_json(output / 'protocol.json', protocol)
    tasks = []

    def save(seed, task):
        profile, X_train, y_train, X_test, y_test, metadata = task
        name = f'arrays/{seed}.npz'
        np.savez_compressed(output / name, X_train=X_train, y_train=y_train,
                            X_test=X_test, y_test=y_test)
        tasks.append({'seed': seed, 'profile': asdict(profile), 'preprocessing': metadata,
                      'arrays': name, 'arrays_sha256': hashlib.sha256((output / name).read_bytes()).hexdigest()})

    for seed in SYNTHETIC_SEEDS:
        save(seed, (*generate_task(seed, split='eval'), {'source': 'crucible_synthetic', 'seed': seed}))
    seed = 90100
    for spec in REAL_DATASETS:
        for split_seed in REAL_SPLITS:
            save(seed, real_task(spec, split_seed))
            seed += 1
    manifest = {'schema': 'frozen-task-pack-v1', 'protocol_sha256': hashlib.sha256(
        (output / 'protocol.json').read_bytes()).hexdigest(), 'tasks': tasks}
    atomic_json(output / 'manifest.json', manifest)
    print({'tasks': len(tasks), 'sources': dict(Counter(t['preprocessing']['source'] for t in tasks)),
           'manifest_sha256': hashlib.sha256((output / 'manifest.json').read_bytes()).hexdigest()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    build(parser.parse_args().output)
