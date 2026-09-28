"""Transactional raw records; JSONL files are rebuildable projections."""
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3


def fingerprint():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob('*.py')):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return {'code_sha256': digest.hexdigest(), 'dependencies': {
        p: importlib.metadata.version(p) for p in ('numpy', 'scikit-learn', 'scipy', 'httpx')}}


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


class RunStore:
    def __init__(self, directory, manifest, *, resume=False):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / '.run.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.root / 'manifest.json'
            if path.exists():
                if not resume:
                    raise ValueError('Output already contains a run; use --resume or a new directory')
                if json.loads(path.read_text()) != manifest:
                    raise ValueError('Resume manifest mismatch: code, dependencies, model or task configuration changed')
            else:
                if resume:
                    raise ValueError('No manifest to resume')
                if any(p.name != '.run.lock' for p in self.root.iterdir()):
                    raise ValueError('Use an empty output directory for a new run')
                atomic_json(path, manifest)
            self.db = sqlite3.connect(self.root / 'records.sqlite3')
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.execute('CREATE TABLE IF NOT EXISTS records (seed INTEGER PRIMARY KEY, body TEXT NOT NULL)')
            self.db.commit()
        except BaseException:
            self.lock.close()
            raise

    def completed(self):
        return {r[0] for r in self.db.execute('SELECT seed FROM records')}

    def save(self, record):
        with self.db:
            self.db.execute('INSERT INTO records VALUES (?, ?)',
                            (record['seed'], json.dumps(record, allow_nan=False)))

    def records(self):
        return [json.loads(r[0]) for r in self.db.execute('SELECT body FROM records ORDER BY seed')]

    def export(self):
        names = ['evaluations.jsonl', 'openjev_ml_nli.jsonl', 'melchior_dpo_pairs.jsonl', 'holdout.jsonl', 'errors.jsonl']
        files = {name: (self.root / (name+'.tmp')).open('w') for name in names}
        def write(name, r):
            files[name].write(json.dumps(r, allow_nan=False) + '\n')
        try:
            for r in self.records():
                write('evaluations.jsonl', r)
                if 'error' in r:
                    write('errors.jsonl', r)
                    continue
                outcome = r['outcome']
                if r['profile']['split'] == 'train' and r['mode'] != 'mock':
                    for row in outcome['nli_records']:
                        write('openjev_ml_nli.jsonl', row)
                    if outcome['dpo_record']:
                        write('melchior_dpo_pairs.jsonl', outcome['dpo_record'])
                if r['profile']['split'] == 'eval' and outcome['winner'] in {'A', 'B'}:
                    write('holdout.jsonl', r)
            for name, f in files.items():
                f.flush(); os.fsync(f.fileno()); f.close()
                (self.root / (name+'.tmp')).replace(self.root / name)
        finally:
            for f in files.values():
                f.close()

    def close(self):
        self.db.close()
        self.lock.close()
