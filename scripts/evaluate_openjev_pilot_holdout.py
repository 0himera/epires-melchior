"""Diagnostic evaluation on historical Crucible eval pairs.

These pairs were inspected during development. The result is not an independent
test and must not be used for checkpoint selection or promotion.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from melchior.crucible.decisions import comparison, nli_rows, symmetric_choice
from scripts.train_openjev_adapter import atomic_json


def records(paths):
    for path in paths:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row=json.loads(line)
            if row.get('schema')!='predictions-v2' or row['profile']['split']!='eval':
                raise ValueError(f'Unexpected schema or split in {path}')
            if row['outcome']['winner'] not in ('A','B'):
                continue
            yield path.name,path.parent.name,digest,row


def main(args):
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    tok.padding_side='right'
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    source=list(records(args.datasets))
    if not source:
        raise ValueError('No decisive eval pairs')
    items=[]
    for filename,cohort,digest,row in source:
        views=[]
        for swapped in (False,True):
            payload=comparison(row['profile'],row['pair'],swapped=swapped,
                               generation_swapped=row.get('generation_swapped',False))
            views.extend(nli_rows(payload,'A'))
        encoded=[tok('Premise: '+v['premise']+'\nHypothesis: '+v['hypothesis'],
                     add_special_tokens=True,truncation=False) for v in views]
        max_tokens=max(len(e['input_ids']) for e in encoded)
        items.append((row,cohort,digest,encoded,max_tokens))
    overlength=[(r['seed'],cohort,length) for r,cohort,_,_,length in items if length>args.max_length]
    if overlength:
        raise ValueError(f'Historical pair exceeds model input budget: {overlength}')

    base=AutoModelForSequenceClassification.from_pretrained(
        args.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    base.config.pad_token_id=tok.pad_token_id
    base.config.get_text_config().pad_token_id=tok.pad_token_id
    base.config.use_cache=False
    base.config.get_text_config().use_cache=False
    if args.adapter:
        model=PeftModel.from_pretrained(base,args.adapter,local_files_only=True).eval()
    else:
        model=base.eval()
    torch.cuda.reset_peak_memory_stats()
    results=[]
    with torch.inference_mode():
        for row,cohort,digest,encoded,max_tokens in items:
            ent=[]
            for e in encoded:
                inputs={k:torch.tensor([v],device='cuda') for k,v in e.items()}
                logits=model(**inputs).logits.float()
                ent.append(logits.softmax(-1)[0,1].item())
            p=ent[0]/max(ent[0]+ent[1],1e-12)
            q=ent[2]/max(ent[2]+ent[3],1e-12)
            symmetric=symmetric_choice(p,q)
            winner=row['outcome']['winner']
            results.append({'cohort':cohort,'seed':row['seed'],
                            'dataset_id':row['profile']['dataset_id'],
                            'metric':row['profile']['metric'],
                            'operator':row['pair']['operator'],'winner':winner,
                            'normal_p_a':p,'swapped_p_a':q,'symmetric':symmetric,
                            'normal_correct':('A' if p>=.5 else 'B')==winner,
                            'swapped_correct':('A' if q>=.5 else 'B')!=winner,
                            'swap_consistent':(p>=.5)!=(q>=.5),
                            'symmetric_correct':symmetric['choice']==winner,
                            'max_tokens':max_tokens})
    aggregate={}
    for cohort in sorted({r['cohort'] for r in results}):
        rows=[r for r in results if r['cohort']==cohort]
        aggregate[cohort]={'n':len(rows),'normal_correct':sum(r['normal_correct'] for r in rows),
                           'swapped_correct':sum(r['swapped_correct'] for r in rows),
                           'symmetric_correct':sum(r['symmetric_correct'] for r in rows),
                           'swap_consistent':sum(r['swap_consistent'] for r in rows),
                           'mean_order_disagreement':sum(r['symmetric']['order_disagreement'] for r in rows)/len(rows),
                           'winner_counts':dict(Counter(r['winner'] for r in rows)),
                           'dataset_groups':len({r['dataset_id'] for r in rows})}
    atomic_json(args.output,{'scope':'historical development data; no checkpoint selection or promotion',
                            'model':str(args.model),'adapter':str(args.adapter) if args.adapter else None,
                            'data_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in args.datasets},
                            'max_length':args.max_length,'max_observed_tokens':max(x[-1] for x in items),
                            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                            'aggregate':aggregate,'results':results})
    print(json.dumps({'aggregate':aggregate,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--adapter',type=Path)
    parser.add_argument('--datasets',type=Path,nargs='+',required=True)
    parser.add_argument('--max-length',type=int,default=2048)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
