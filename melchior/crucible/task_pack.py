"""Load frozen inputs independently of candidate generation and model scores."""
import hashlib
import json
from pathlib import Path

import numpy as np

from melchior.crucible.environments import TaskProfile


class TaskPack:
    def __init__(self, directory, *, split):
        self.root = Path(directory).resolve()
        path = self.root / 'manifest.json'
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = json.loads(path.read_text())
        if manifest['schema'] != 'frozen-task-pack-v1':
            raise ValueError('Unknown task pack schema')
        self.tasks = {}
        for row in manifest['tasks']:
            seed = row['seed']
            if not isinstance(seed, int) or seed < 0 or seed in self.tasks:
                raise ValueError('Task pack seeds must be unique nonnegative integers')
            if row['profile']['split'] != split:
                raise ValueError('Task pack split mismatch')
            file = (self.root / row['arrays']).resolve()
            if not file.is_relative_to(self.root):
                raise ValueError('Task arrays must be inside the frozen pack')
            if hashlib.sha256(file.read_bytes()).hexdigest() != row['arrays_sha256']:
                raise ValueError(f'Task array checksum mismatch: {seed}')
            self.tasks[seed] = row
        self.seeds = tuple(self.tasks)

    def load(self, seed):
        row = self.tasks[seed]
        path = self.root / row['arrays']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['arrays_sha256']:
            raise ValueError(f'Task array checksum mismatch: {seed}')
        with np.load(path, allow_pickle=False) as source:
            arrays = tuple(source[key] for key in ('X_train', 'y_train', 'X_test', 'y_test'))
        return (TaskProfile(**row['profile']), *arrays)

    def metadata(self, seed):
        row = self.tasks[seed]
        return {'pack_sha256': self.sha256, 'arrays': row['arrays'],
                'arrays_sha256': row['arrays_sha256'], 'preprocessing': row['preprocessing']}
