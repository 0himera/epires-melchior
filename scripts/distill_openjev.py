"""Finite, text-only OpenJev distillation from a frozen trained NLI teacher.

Export stores all three logits, keyed by exact input text, before student
training. Train/dev remain grouped; the independent test is never read.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from scripts.train_openjev_adapter import (
    atomic_json, batch, encode_rows, evaluate, formatted, load_model, read_rows,
)


def row_id(row):
    return hashlib.sha256(formatted(row).encode()).hexdigest()


def mixed_loss(student, teacher, labels, temperature=2.0, soft_weight=0.7):
    if temperature <= 0 or not 0 <= soft_weight <= 1:
        raise ValueError('Invalid temperature or soft weight')
    soft = F.kl_div(
        F.log_softmax(student.float() / temperature, dim=-1),
        F.softmax(teacher.float() / temperature, dim=-1),
        reduction='batchmean',
    ) * temperature ** 2
    hard = F.cross_entropy(student.float(), labels)
    return soft_weight * soft + (1 - soft_weight) * hard, soft, hard


def datasets(directory):
    ml = {s: read_rows(directory / f'ml_{s}.jsonl') for s in ('train', 'dev')}
    replay = {s: read_rows(directory / f'replay_{s}.jsonl') for s in ('train', 'dev')}
    for kind in (ml, replay):
        if {r['group'] for r in kind['train']} & {r['group'] for r in kind['dev']}:
            raise ValueError('Train/dev group overlap')
    flat = {}
    for split in ('train', 'dev'):
        flat[f'ml_{split}'] = [r for g in ml[split] for r in g['views']]
        flat[f'replay_{split}'] = replay[split]
    if {row_id(r) for k, v in flat.items() if k.endswith('train') for r in v} & {
        row_id(r) for k, v in flat.items() if k.endswith('dev') for r in v
    }:
        raise ValueError('Exact train/dev input overlap')
    return ml, replay, flat


def file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def data_hashes(directory):
    return {p.name: file_hash(p) for p in sorted(directory.glob('*.jsonl'))}


def configure(model, tok):
    model.config.pad_token_id = tok.pad_token_id
    model.config.get_text_config().pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    model.config.get_text_config().use_cache = False
    labels = {int(k): v for k, v in model.config.id2label.items()}
    if labels != {0: 'contradiction', 1: 'entailment', 2: 'neutral'}:
        raise ValueError(f'Unexpected class mapping: {labels}')


def tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tok.padding_side = 'right'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def teacher_export(args):
    args.output.mkdir(parents=True, exist_ok=False)
    _, _, flat = datasets(args.data)
    tok = tokenizer(args.model)
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation='sdpa',
    ).to('cuda')
    configure(base, tok)
    model = PeftModel.from_pretrained(base, args.teacher_adapter, local_files_only=True).eval()
    manifest = {
        'status': 'exporting', 'teacher_base': args.model,
        'teacher_adapter': str(args.teacher_adapter),
        'teacher_adapter_sha256': file_hash(args.teacher_adapter / 'adapter_model.safetensors'),
        'data_sha256': data_hashes(args.data), 'max_length': args.max_length,
        'label_mapping': {'contradiction': 0, 'entailment': 1, 'neutral': 2},
        'torch': torch.__version__, 'rows': {},
        'scope': 'Existing grouped train/dev only; no independent test',
    }
    atomic_json(args.output / 'manifest.json', manifest)
    started = time.monotonic()
    exported = {}
    with torch.inference_mode():
        for name, rows in flat.items():
            encoded, _ = encode_rows(tok, rows, args.max_length, strict=True)
            path = args.output / f'{name}.jsonl'
            with path.open('w') as stream:
                for start in range(0, len(rows), args.batch_rows):
                    inputs, _ = batch(tok, encoded[start:start + args.batch_rows])
                    logits = model(**inputs).logits.float()
                    if not torch.isfinite(logits).all():
                        raise RuntimeError('Nonfinite teacher logits')
                    for row, values in zip(rows[start:start + args.batch_rows], logits.cpu().tolist()):
                        key = row_id(row)
                        if key in exported and exported[key]['label'] != row['label']:
                            raise ValueError('Identical source inputs have conflicting labels')
                        record = exported.setdefault(key, {'row_id': key, 'label': row['label'], 'logits': values})
                        stream.write(json.dumps(record) + '\n')
                    stream.flush()
                    if start % 128 == 0:
                        print(json.dumps({'phase': 'teacher', 'split': name,
                                          'rows': min(start + args.batch_rows, len(rows)),
                                          'total': len(rows), 'elapsed_s': time.monotonic() - started}), flush=True)
            manifest['rows'][name] = len(rows)
            manifest.setdefault('logits_sha256', {})[path.name] = file_hash(path)
            atomic_json(args.output / 'manifest.json', manifest)
    manifest.update(status='complete', elapsed_s=time.monotonic() - started)
    atomic_json(args.output / 'manifest.json', manifest)
    print('TEACHER_EXPORT_COMPLETE', flush=True)


def cached_logits(directory, data):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['status'] != 'complete' or manifest['data_sha256'] != data_hashes(data):
        raise ValueError('Teacher cache is incomplete or uses different data')
    targets = {}
    for name, checksum in manifest['logits_sha256'].items():
        path = directory / name
        if file_hash(path) != checksum:
            raise ValueError('Teacher cache hash mismatch')
        for row in read_rows(path):
            key = row['row_id']
            if key in targets and targets[key] != row:
                raise ValueError('Conflicting teacher logits')
            targets[key] = row
    return targets, manifest


def teacher_batch(rows, targets):
    values = []
    for row in rows:
        target = targets[row_id(row)]
        if row['label'] != target['label']:
            raise ValueError('Cached label differs from source')
        values.append(target['logits'])
    return torch.tensor(values, dtype=torch.float32, device='cuda')


@torch.inference_mode()
def fidelity(model, tok, ml, encoded, replay, replay_encoded, targets, temperature):
    """Measure held-out soft-distribution agreement, separately from gold accuracy."""
    model.eval()
    result = {}
    for name, rows, tokens in (
        ('ml', [r for g in ml for r in g['views']], [r for g in encoded for r in g]),
        ('replay', replay, replay_encoded),
    ):
        total_kl = total_agreement = teacher_correct = student_correct = 0
        for start in range(0, len(rows), 8):
            inputs, labels = batch(tok, tokens[start:start + 8])
            student = model(**inputs).logits.float()
            teacher = teacher_batch(rows[start:start + 8], targets)
            kl = F.kl_div(F.log_softmax(student / temperature, -1),
                          F.softmax(teacher / temperature, -1), reduction='sum')
            total_kl += kl.item() * temperature ** 2
            total_agreement += (student.argmax(-1) == teacher.argmax(-1)).sum().item()
            teacher_correct += (teacher.argmax(-1) == labels).sum().item()
            student_correct += (student.argmax(-1) == labels).sum().item()
        result[name] = {'rows': len(rows), 'mean_kl_t2': total_kl / len(rows),
                        'teacher_student_agreement': total_agreement / len(rows),
                        'teacher_nli_accuracy': teacher_correct / len(rows),
                        'student_nli_accuracy': student_correct / len(rows)}
    return result


def save_checkpoint(model, output, name, step):
    temporary = output / f'.{name}.saving'
    model.save_pretrained(temporary)
    atomic_json(temporary / 'training_state.json', {'step': step, 'adapter_only': True})
    temporary.rename(output / name)
    print(json.dumps({'phase': 'saved', 'checkpoint': name, 'step': step}), flush=True)


def train(args):
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    targets, teacher_manifest = cached_logits(args.teacher_cache, args.data)
    ml, replay, _ = datasets(args.data)
    tok = tokenizer(args.model)
    tok.save_pretrained(args.output / 'tokenizer')
    encoded = {s: [encode_rows(tok, g['views'], args.max_length)[0] for g in ml[s]]
               for s in ('train', 'dev')}
    replay_encoded = {s: encode_rows(tok, replay[s], args.max_length, strict=True)[0]
                      for s in ('train', 'dev')}
    model, modules = load_model(args.model, tok, True)
    configure(model, tok)
    parameters = [p for p in model.parameters() if p.requires_grad]
    atomic_json(args.output / 'manifest.json', {
        'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'data_sha256': data_hashes(args.data), 'teacher': teacher_manifest,
        'trainable_parameters': sum(p.numel() for p in parameters), 'target_modules': modules,
        'objective': f'Equal ML/replay weight; {args.soft_weight} KL(teacher||student)*T^2 '
                     f'+ {1 - args.soft_weight} gold CE; T={args.temperature}',
        'ml_groups': {s: len(ml[s]) for s in ml},
        'replay_rows': {s: len(replay[s]) for s in replay},
        'scope': 'One epoch, all training rows once; existing dev; independent test untouched',
    })
    initial = evaluate(model, tok, ml['dev'], encoded['dev'], replay_encoded['dev'])
    initial['distillation_fidelity'] = fidelity(
        model, tok, ml['dev'], encoded['dev'], replay['dev'], replay_encoded['dev'], targets, args.temperature)
    atomic_json(args.output / 'baseline.json', initial)
    print('BASELINE', json.dumps({k: v for k, v in initial.items() if k != 'results'}), flush=True)
    model.train()
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.01)
    order = list(range(len(ml['train'])))
    random.Random(args.seed).shuffle(order)
    replay_order = list(range(len(replay['train'])))
    random.Random(args.seed + 1000).shuffle(replay_order)
    micro = args.micro_groups
    batches = math.ceil(len(order) / micro)
    total_steps = math.ceil(batches / args.accumulation)
    step = 0
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    seen_ml = seen_replay = 0
    completed_epoch = True
    for bi, start in enumerate(range(0, len(order), micro)):
        window_start = bi // args.accumulation * args.accumulation
        window_end = min(batches, window_start + args.accumulation)
        group_count = min(len(order), window_end * micro) - window_start * micro
        replay_start = bi * len(replay_order) // batches
        replay_end = (bi + 1) * len(replay_order) // batches
        window_replay_count = (window_end * len(replay_order) // batches
                               - window_start * len(replay_order) // batches)
        indices = order[start:start + micro]
        ml_rows = [row for i in indices for row in ml['train'][i]['views']]
        inputs, labels = batch(tok, [row for i in indices for row in encoded['train'][i]])
        ml_loss, soft, hard = mixed_loss(model(**inputs).logits, teacher_batch(ml_rows, targets),
                                        labels, args.temperature, args.soft_weight)
        (.5 * ml_loss * len(indices) / group_count).backward()
        replay_indices = replay_order[replay_start:replay_end]
        rows = [replay['train'][i] for i in replay_indices]
        inputs, labels = batch(tok, [replay_encoded['train'][i] for i in replay_indices])
        replay_loss, _, _ = mixed_loss(model(**inputs).logits, teacher_batch(rows, targets),
                                       labels, args.temperature, args.soft_weight)
        (.5 * replay_loss * len(rows) / window_replay_count).backward()
        seen_ml += len(ml_rows)
        seen_replay += len(rows)
        if bi + 1 == window_end:
            step += 1
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm):
                raise RuntimeError('Nonfinite student gradient')
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            progress = {'phase': 'train', 'step': step, 'total_steps': total_steps,
                        'ml_loss': ml_loss.item(), 'replay_loss': replay_loss.item(),
                        'soft_kl': soft.item(), 'hard_ce': hard.item(), 'gradient_norm': norm.item(),
                        'elapsed_s': time.monotonic() - started,
                        'seen_ml_rows': seen_ml, 'seen_replay_rows': seen_replay}
            atomic_json(args.output / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
            if step == 1 or step % args.save_every == 0:
                save_checkpoint(model, args.output, f'step_{step:04d}', step)
            if args.stop_at_utc and time.time() >= args.stop_at_utc:
                completed_epoch = bi + 1 == batches
                break
    save_checkpoint(model, args.output, 'final', step)
    metrics = evaluate(model, tok, ml['dev'], encoded['dev'], replay_encoded['dev'])
    metrics['distillation_fidelity'] = fidelity(
        model, tok, ml['dev'], encoded['dev'], replay['dev'], replay_encoded['dev'], targets, args.temperature)
    atomic_json(args.output / 'final' / 'evaluation.json', metrics)
    # Verify the saved adapter in a fresh model, after releasing optimizer state.
    reference = metrics['results'][:4]
    del optimizer, parameters, model
    gc.collect()
    torch.cuda.empty_cache()
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to('cuda')
    configure(base, tok)
    reloaded = PeftModel.from_pretrained(base, args.output / 'final', local_files_only=True)
    verification = evaluate(reloaded, tok, ml['dev'][:4], encoded['dev'][:4], replay_encoded['dev'][:8])
    error = max(abs(a[k] - b[k]) for a, b in zip(reference, verification['results'])
                for k in ('p_a', 'reverse_p_a'))
    report = {'status': 'complete' if completed_epoch else 'time_limited',
              'steps': step, 'total_steps': total_steps, 'seen_ml_rows': seen_ml,
              'seen_replay_rows': seen_replay, 'reload_verified': error < 1e-4,
              'reload_max_score_error': error,
              'baseline': {k: v for k, v in initial.items() if k != 'results'},
              'final': {k: v for k, v in metrics.items() if k != 'results'},
              'elapsed_s': time.monotonic() - started,
              'scope': 'Development comparison only, no independent quality claim or promotion'}
    atomic_json(args.output / 'summary.json', report)
    print('DISTILLATION_COMPLETE', json.dumps(report), flush=True)
    if not report['reload_verified']:
        raise RuntimeError('Saved student adapter failed reload verification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['export', 'train'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--teacher-adapter', type=Path)
    parser.add_argument('--teacher-cache', type=Path)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--batch-rows', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--micro-groups', type=int, default=2)
    parser.add_argument('--accumulation', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--temperature', type=float, default=2.)
    parser.add_argument('--soft-weight', type=float, default=.7)
    parser.add_argument('--save-every', type=int, default=4)
    parser.add_argument('--stop-at-utc', type=float)
    args = parser.parse_args()
    if min(args.micro_groups, args.accumulation, args.save_every, args.batch_rows) < 1:
        parser.error('Batch sizes, accumulation and save interval must be positive')
    if args.mode == 'export' and args.teacher_adapter is None:
        parser.error('export requires --teacher-adapter')
    if args.mode == 'train' and args.teacher_cache is None:
        parser.error('train requires --teacher-cache')
    teacher_export(args) if args.mode == 'export' else train(args)
