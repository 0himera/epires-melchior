"""Prepare grouped ML rotations and source NLI replay without model inference."""
import argparse
from collections import defaultdict, Counter
import gzip
import hashlib
import heapq
import json
from pathlib import Path
import random

from melchior.crucible.decisions import comparison, normalize_pair, nli_rows


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))


def prepare_ml(source, details, output):
    accepted={r['seed'] for r in json.loads(details.read_text()) if r['winner']==r['winner_5000']}
    records=[json.loads(l) for l in source.read_text().splitlines() if l.strip()]
    records=[r for r in records if r['seed'] in accepted]
    by_group=defaultdict(list)
    for r in records:
        by_group[r['profile']['dataset_id']].append(r)
    # Stratify whole groups by metric/operator. No pair or rotation crosses splits.
    strata=defaultdict(list)
    for group, rs in by_group.items():
        strata[(rs[0]['profile']['metric'],rs[0]['pair']['operator'])].append(group)
    dev_groups=set()
    for key, groups in sorted(strata.items()):
        ordered=sorted(groups,key=lambda g:digest('split42:'+g))
        n=min(len(groups)-1,max(1,round(len(groups)*.2))) if len(groups)>1 else 0
        dev_groups.update(ordered[:n])
    splits={'train':[],'dev':[]}
    for r in records:
        pair=normalize_pair(r['pair'],generation_swapped=r.get('generation_swapped',False))
        winner=r['outcome']['winner']
        views=[]
        for swapped in (False,True):
            label=('B' if winner=='A' else 'A') if swapped else winner
            views.extend(nli_rows(comparison(r['profile'],pair,swapped=swapped),label))
        group=r['profile']['dataset_id']
        splits['dev' if group in dev_groups else 'train'].append({
            'seed':r['seed'],'group':group,'metric':r['profile']['metric'],
            'operator':pair['operator'],'winner':winner,'views':views,
            'candidate_ids':[digest(pair['code_'+k]) for k in ('a','b')]})
    for split,groups in splits.items():
        write_rows(output/f'ml_{split}.jsonl',groups)
    assert not ({r['group'] for r in splits['train']} & {r['group'] for r in splits['dev']})
    return {k:{'groups':len(v),'rows':4*len(v),'operators':dict(Counter(r['operator'] for r in v))} for k,v in splits.items()}


def prepare_replay(sources, output):
    # Bounded, deterministic reservoirs per source/class/split. Split by premise,
    # never by individual hypothesis/vote. Replay dev measures retention, not novelty.
    heaps=defaultdict(list)
    scanned=Counter()
    index=0
    for name in ('text','ifcomplex'):
        with gzip.open(sources/(name+'.jsonl.gz'),'rt') as stream:
            for line in stream:
                r=json.loads(line);scanned[name]+=1
                if r.get('image') or r['label'] not in (0,1,2):continue
                if len(r['premise'])+len(r['hypothesis'])>6000:continue
                group=digest(' '.join(r['premise'].split()))
                split='dev' if int(group[:8],16)%5==0 else 'train'
                quota=(500 if split=='train' else 125) if name=='text' else (250 if split=='train' else 63)
                key=(name,split,r['label'])
                priority=int(digest(str(index)+line)[:16],16)
                row={k:r.get(k,'') for k in ('premise','hypothesis','label','source','image')}
                row['group']=group
                item=(-priority,index,row)
                if len(heaps[key])<quota:heapq.heappush(heaps[key],item)
                elif priority < -heaps[key][0][0]:heapq.heapreplace(heaps[key],item)
                index+=1
    splits={split:[x[2] for key,heap in sorted(heaps.items()) if key[1]==split for x in sorted(heap)] for split in ('train','dev')}
    assert not ({r['group'] for r in splits['train']} & {r['group'] for r in splits['dev']})
    for split,rows in splits.items():write_rows(output/f'replay_{split}.jsonl',rows)
    return {'scanned':dict(scanned),'splits':{k:{'rows':len(v),'labels':dict(Counter(r['label'] for r in v))} for k,v in splits.items()}}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--details',type=Path,required=True)
    p.add_argument('--replay-sources',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report={'ml':prepare_ml(args.source,args.details,args.output)}
    if args.replay_sources:report['replay']=prepare_replay(args.replay_sources,args.output)
    report['source_sha256']=hashlib.sha256(args.source.read_bytes()).hexdigest()
    report['details_sha256']=hashlib.sha256(args.details.read_bytes()).hexdigest()
    report['replay_revision']='98ddc2bba16930bc975b711bb3bca269dce4459c'
    (args.output/'data_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
