"""Server-side supervisor for the finite pilot; restores Qwen after completion.

Runs on the host with Docker access. Reuses an explicitly named training
container's image/devices/mounts; never promotes any adapter into serving.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def inspect(name):
    return json.loads(subprocess.check_output(['docker','inspect',name]))[0]


def wait(name):
    return int(subprocess.check_output(['docker','wait',name],text=True).strip())


def create(template,name,command,repo):
    # Use Docker CLI with explicit devices and volumes from the known pilot.
    original=inspect(template)
    mounts=[]
    for m in original['Mounts']:
        src=repo if m['Destination']=='/repo' else m['Source']
        mounts.extend(['-v',src+':'+m['Destination']+('' if m['RW'] else ':ro')])
    cmd=['docker','run','-d','--name',name,'--device','/dev/kfd','--device','/dev/dri',
         '--group-add','video','--ipc','host','--security-opt','seccomp=unconfined','--network','none',
         '-e','PYTHONPATH=/repo','-e','OMP_NUM_THREADS=4','-e','TOKENIZERS_PARALLELISM=false',
         '-e','HF_HUB_OFFLINE=1',*mounts,'--entrypoint','python3',original['Image'],'-u',*command]
    subprocess.run(cmd,check=True)


def report(root,status,errors):
    runs={}
    for p in sorted((root/'runs').glob('*/summary.json')):
        s=json.loads(p.read_text());base=json.loads((p.parent/'baseline.json').read_text())
        selected=next((x for x in s['history'] if x['epoch']==s['selected_epoch']),None)
        verification=p.parent/'reload_verification.json'
        runs[p.parent.name]={'baseline':{k:v for k,v in base.items() if k!='results'},
                             'selected':selected,'history':s['history'],
                             'reload_verification':json.loads(verification.read_text()) if verification.exists() else None}
    value={'status':status,'runs':runs,'errors':errors,'evaluation_scope':'development only; no independent test or promotion',
           'updated_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    tmp=root/'pilot_report.tmp';tmp.write_text(json.dumps(value,indent=2));tmp.replace(root/'pilot_report.json')


def main(args):
    errors=[]
    try:
        # Complete the already-started seed-42 consistency run first.
        jobs=[(args.first,'s42_c01',42,.1)]
        if wait(args.first)!=0:
            raise RuntimeError('Initial training container failed; inspect its logs')
        for seed,weight in [(42,0.),(43,.1),(43,0.)]:
            jobs.append((f'openjev-lora-pilot-s{seed}-c{int(weight*10)}',f's{seed}_c{int(weight*10):02d}',seed,weight))
        for i,(name,run,seed,weight) in enumerate(jobs):
            if i:
                create(args.first,name,['/repo/scripts/train_openjev_adapter.py','--model','/model/qwen3.5-4b-nli-v5',
                    '--data','/experiment/data','--output',f'/experiment/runs/{run}','--seed',str(seed),
                    '--consistency',str(weight),'--epochs','3'],str(args.repository))
                if wait(name)!=0:raise RuntimeError(f'Training failed: {name}')
            verify=name+'-verify'
            create(args.first,verify,['/repo/scripts/verify_openjev_adapter.py','--model','/model/qwen3.5-4b-nli-v5',
                '--data','/experiment/data','--run',f'/experiment/runs/{run}'],str(args.repository))
            if wait(verify)!=0:raise RuntimeError(f'Adapter reload failed: {verify}')
            report(args.root,'running',errors)
        report(args.root,'complete',errors)
    except Exception as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
        report(args.root,'failed',errors)
        raise
    finally:
        # The experiment borrowed GPU memory from the idle generator service.
        subprocess.run(['docker','start','vllm-qwen'],check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--first',required=True)
    p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    main(p.parse_args())
