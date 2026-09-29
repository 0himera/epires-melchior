"""Measure identical optimizer windows with several microbatch/checkpoint settings.

Dropout is disabled only for gradient comparisons. Timings use training dropout,
CUDA synchronization, one warmup window and repeated forward/backward windows.
No optimizer updates or changes to the base checkpoint are made.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, set_seed

from scripts.train_openjev_adapter import (
    atomic_json, batch, encode_rows, load_model, probabilities, read_rows,
)


def backward_window(model, tok, groups, replay, micro, consistency=.1):
    model.zero_grad(set_to_none=True)
    loss_total=0.
    for start in range(0,len(groups),micro):
        selected=groups[start:start+micro]
        rows=[r for group in selected for r in group]
        weight=len(selected)/len(groups)
        inputs,labels=batch(tok,rows)
        logits=model(**inputs).logits.float()
        p,q=probabilities(logits)
        loss=(.5*F.cross_entropy(logits,labels)+consistency*((p+q-1)**2).mean())*weight
        loss.backward();loss_total+=loss.item()
        inputs,labels=batch(tok,replay[start*4:(start+len(selected))*4])
        loss=.5*F.cross_entropy(model(**inputs).logits.float(),labels)*weight
        loss.backward();loss_total+=loss.item()
    return loss_total


def main(args):
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    tok.padding_side='right'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    raw=read_rows(args.data/'ml_train.jsonl')
    order=list(range(len(raw)));random.Random(44).shuffle(order)
    groups=[encode_rows(tok,raw[i]['views'],2048)[0] for i in order[:8]]
    replay,_=encode_rows(tok,read_rows(args.data/'replay_train.jsonl'),2048,strict=False)
    random.Random(1044).shuffle(replay);replay=replay[:32]
    parts=[item.split(':') for item in args.configs.split(',')]
    if any(len(p)!=2 or p[1] not in ('on','off') for p in parts):
        raise ValueError('Each configuration must be micro-groups:on or micro-groups:off')
    configs=[(int(count),checkpoint=='on') for count,checkpoint in parts]
    if configs[0]!=(2,True) or any(m not in (1,2,4,8) for m,_ in configs):
        raise ValueError('Start with 2:on; micro groups must divide the effective batch of 8')
    reference={};results=[]
    for index,(micro,checkpoint) in enumerate(configs):
        set_seed(43)
        model,_=load_model(args.model,tok,checkpoint)
        model.train()
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        # Include Adam state in the memory measurement, even though no update is made.
        adam_state=[torch.zeros_like(p) for _,p in trainable for _ in range(2)]
        drops=[(m,m.p) for m in model.modules() if isinstance(m,torch.nn.Dropout)]
        for m,_ in drops:m.p=0.
        comparisons={}
        for size in (8,6):  # Full batch and the actual final, incomplete epoch batch.
            loss=backward_window(model,tok,groups[:size],replay[:size*4],micro)
            grad=torch.cat([p.grad.detach().float().flatten().cpu() for _,p in trainable])
            if not torch.isfinite(grad).all():raise RuntimeError('Nonfinite gradient')
            if index==0:reference[size]=(loss,grad)
            base_loss,base_grad=reference[size]
            g64,b64=grad.double(),base_grad.double()
            cosine=(torch.dot(g64,b64)/(g64.norm()*b64.norm())).item()
            rel=(g64-b64).norm().item()/b64.norm().item()
            del g64,b64
            comparisons[str(size)]={'loss':loss,'reference_loss':base_loss,'gradient_cosine':cosine,'gradient_relative_error':rel}
            if cosine < .999 or rel > .03 or abs(loss-base_loss) > .01:
                atomic_json(args.output,{'results':results,'complete':False,
                            'rejected':{'micro_groups':micro,'gradient_checkpointing':checkpoint,
                                        'reason':'gradient comparison failed','comparisons':comparisons}})
                raise RuntimeError(f'Gradient comparison failed: {comparisons}')
            del grad
        for m,p in drops:m.p=p
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        durations=[]
        for iteration in range(args.repeats+1):
            torch.cuda.synchronize();started=time.monotonic()
            backward_window(model,tok,groups,replay,micro)
            torch.cuda.synchronize()
            if iteration:durations.append(time.monotonic()-started)
        result={'micro_groups':micro,'accumulation':8//micro,'gradient_checkpointing':checkpoint,
                'seconds':durations,'median_seconds':statistics.median(durations),
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                'gradient_comparisons':comparisons}
        results.append(result)
        atomic_json(args.output,{'results':results,'complete':len(results)==len(configs),
                    'torch':torch.__version__,
                    'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data.glob('*train.jsonl')},
                    'timing_scope':'forward/backward only, dropout enabled, one warmup; Adam state memory included',
                    'ml_group_indices':order[:8],
                    'ml_max_tokens':max(len(r['input_ids']) for g in groups for r in g),
                    'replay_max_tokens':max(len(r['input_ids']) for r in replay)})
        print(json.dumps(result),flush=True)
        del model,trainable,adam_state,drops,m,p
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--configs',default='2:on,4:on,8:on',help='Comma-separated micro-groups:checkpoint on/off')
    main(parser.parse_args())
