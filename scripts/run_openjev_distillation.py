"""Host-side finite distillation runner; records logs and restores Qwen."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def write_status(root, **values):
    values['updated_at_utc'] = datetime.now(timezone.utc).isoformat()
    temporary = root / 'status.tmp'
    temporary.write_text(json.dumps(values, indent=2))
    temporary.replace(root / 'status.json')


def execute(args, mode, command):
    name = args.prefix + '-' + mode
    docker = [
        'docker', 'run', '-d', '--name', name, '--device', '/dev/kfd', '--device', '/dev/dri',
        '--group-add', 'video', '--ipc', 'host', '--security-opt', 'seccomp=unconfined',
        '--network', 'none', '-e', 'PYTHONPATH=/repo', '-e', 'OMP_NUM_THREADS=4',
        '-e', 'TOKENIZERS_PARALLELISM=false', '-e', 'HF_HUB_OFFLINE=1',
        '-v', f'{args.root}/repo:/repo:ro', '-v', f'{args.root}:/experiment',
        '-v', f'{args.data}:/training:ro', '-v', f'{args.models}:/model:ro',
        '-v', f'{args.adapter}:/adapter:ro', '--entrypoint', 'python3', args.image,
        '-u', '/repo/scripts/distill_openjev.py', *command,
    ]
    (args.root / f'{mode}-command.json').write_text(json.dumps(docker, indent=2))
    subprocess.run(docker, check=True)
    with (args.root / f'{mode}.log').open('w') as log:
        logger = subprocess.Popen(['docker', 'logs', '-f', name], stdout=log, stderr=log)
        try:
            seconds = max(1., args.hard_stop_utc - time.time())
            result = subprocess.check_output(['docker', 'wait', name], text=True, timeout=seconds)
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'stop', '-t', '10', name], check=True)
            raise RuntimeError(f'Hard deadline reached during {mode}')
        finally:
            try:
                logger.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.terminate()
                logger.wait(timeout=10)
    if int(result.strip()) != 0:
        raise RuntimeError(f'{mode} container failed: exit {result.strip()}')


def main(args):
    try:
        subprocess.run(['docker', 'stop', '-t', '10', 'vllm-qwen'], check=True)
        write_status(args.root, status='running', phase='teacher_export')
        execute(args, 'export', ['export', '--model', '/model/qwen3.5-4b-nli-v5',
            '--teacher-adapter', '/adapter', '--data', '/training',
            '--output', '/experiment/teacher', '--batch-rows', '8'])
        write_status(args.root, status='running', phase='student_training')
        execute(args, 'train', ['train', '--model', '/model/qwen3.5-0.8b-nli-v2s-long',
            '--teacher-cache', '/experiment/teacher', '--data', '/training',
            '--output', '/experiment/student', '--stop-at-utc', str(args.train_stop_utc)])
        write_status(args.root, status='complete', phase='finished',
                     summary=json.loads((args.root / 'student/summary.json').read_text()))
    except Exception as exc:
        write_status(args.root, status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        subprocess.run(['docker', 'start', 'vllm-qwen'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--prefix', default='openjev-distill-20260930')
    parser.add_argument('--train-stop-utc', type=float, required=True)
    parser.add_argument('--hard-stop-utc', type=float, required=True)
    main(parser.parse_args())
