"""Grouped LoRA pilot with replay, paired orientations and per-epoch evaluation.

Requires the server's torch/transformers/peft environment. Never overwrites the
base checkpoint or promotes an adapter into the running service.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from melchior.crucible.decisions import symmetric_choice


def read_rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def atomic_json(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def formatted(row):
    return 'Premise: '+row['premise']+'\nHypothesis: '+row['hypothesis']


def encode_rows(tok, rows, max_length, *, strict=True):
    result=[];discarded=0
    for row in rows:
        encoded=tok(formatted(row),truncation=False,add_special_tokens=True)
        if len(encoded['input_ids']) > max_length:
            if strict:raise ValueError('ML input exceeds max_length; refusing truncation')
            discarded+=1;continue
        result.append({'input_ids':encoded['input_ids'],'attention_mask':encoded['attention_mask'],'labels':row['label']})
    return result,discarded


def batch(tok,rows):
    inputs=tok.pad([{k:v for k,v in r.items() if k!='labels'} for r in rows],padding=True,return_tensors='pt')
    return {k:v.to('cuda') for k,v in inputs.items()},torch.tensor([r['labels'] for r in rows],device='cuda')


def probabilities(logits):
    ent=logits.float().softmax(-1).reshape(-1,4,3)[:,:,1]
    return ent[:,0]/ent[:,:2].sum(1).clamp_min(1e-12),ent[:,2]/ent[:,2:].sum(1).clamp_min(1e-12)


@torch.inference_mode()
def evaluate(model,tok,ml,encoded_ml,replay,batch_groups=2):
    model.eval();results=[]
    for start in range(0,len(ml),batch_groups):
        groups=ml[start:start+batch_groups]
        rows=[x for g in encoded_ml[start:start+batch_groups] for x in g]
        inputs,labels=batch(tok,rows)
        logits=model(**inputs).logits
        p,q=probabilities(logits)
        for group,a,b in zip(groups,p.cpu().tolist(),q.cpu().tolist()):
            sym=symmetric_choice(a,b)
            normal='A' if a>=.5 else 'B'
            reverse='A' if b>=.5 else 'B'
            results.append({'seed':group['seed'],'group':group['group'],'metric':group['metric'],
                            'operator':group['operator'],'winner':group['winner'],
                            'p_a':a,'reverse_p_a':b,'correct':normal==group['winner'],
                            'swapped_correct':reverse!=group['winner'],'swap_consistent':normal!=reverse,
                            'symmetric':sym,'symmetric_correct':sym['choice']==group['winner']})
    predictions=[];labels_all=[];ce_sum=0.
    for start in range(0,len(replay),8):
        inputs,labels=batch(tok,replay[start:start+8])
        logits=model(**inputs).logits.float()
        ce_sum+=F.cross_entropy(logits,labels,reduction='sum').item()
        predictions.extend(logits.argmax(-1).cpu().tolist());labels_all.extend(labels.cpu().tolist())
    n=len(results)
    correct=np.equal(predictions,labels_all)
    return {'ml_pairs':n,'accuracy':sum(r['correct'] for r in results)/n,
            'swapped_accuracy':sum(r['swapped_correct'] for r in results)/n,
            'swap_consistency':sum(r['swap_consistent'] for r in results)/n,
            'symmetric_accuracy':sum(r['symmetric_correct'] for r in results)/n,
            'symmetric_coverage':sum(r['symmetric']['choice']!='UNCERTAIN' for r in results)/n,
            'mean_order_disagreement':sum(r['symmetric']['order_disagreement'] for r in results)/n,
            'always_a_accuracy':sum(r['winner']=='A' for r in results)/n,
            'always_b_accuracy':sum(r['winner']=='B' for r in results)/n,
            'replay_rows':len(replay),'replay_accuracy':float(correct.mean()),'replay_loss':ce_sum/len(replay),
            'replay_accuracy_by_label':{str(k):float(correct[np.array(labels_all)==k].mean()) for k in sorted(set(labels_all))},
            'results':results}


def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    set_seed(args.seed)
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    tok.padding_side='right'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    ml={k:read_rows(args.data/f'ml_{k}.jsonl') for k in ('train','dev')}
    assert not ({x['group'] for x in ml['train']} & {x['group'] for x in ml['dev']})
    encoded={k:[encode_rows(tok,g['views'],args.max_length)[0] for g in v] for k,v in ml.items()}
    replay={};discarded={}
    raw_replay={k:read_rows(args.data/f'replay_{k}.jsonl') for k in ('train','dev')}
    assert not ({x['group'] for x in raw_replay['train']} & {x['group'] for x in raw_replay['dev']})
    for k in ('train','dev'):
        replay[k],discarded[k]=encode_rows(tok,raw_replay[k],args.max_length,strict=False)
        assert set(r['labels'] for r in replay[k])=={0,1,2}
    print('Loading base model',flush=True)
    model=AutoModelForSequenceClassification.from_pretrained(args.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    model.config.pad_token_id=tok.pad_token_id
    model.config.get_text_config().pad_token_id=tok.pad_token_id
    model.config.use_cache=False
    model.config.get_text_config().use_cache=False
    targets={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj',
             'in_proj_qkv','in_proj_z','in_proj_a','in_proj_b','out_proj'}
    modules=[name for name,module in model.named_modules() if isinstance(module,torch.nn.Linear)
             and name.rsplit('.',1)[-1] in targets and not any(s in name.lower() for s in ('visual','vision'))]
    if not modules:raise RuntimeError('No text LoRA targets found')
    config=LoraConfig(task_type=TaskType.SEQ_CLS,r=16,lora_alpha=32,lora_dropout=.05,
                      target_modules=modules,modules_to_save=['score'])
    model=get_peft_model(model,config)
    for name,param in model.named_parameters():
        if any(s in name.lower() for s in ('visual','vision')):param.requires_grad=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    assert any('score' in n for n,p in trainable)
    assert not any('visual' in n or 'vision' in n for n,p in trainable)
    manifest={'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.data.glob('*.jsonl')},
              'torch':torch.__version__,'trainable_parameters':sum(p.numel() for n,p in trainable),
              'trainable_names':[n for n,p in trainable],'target_modules':modules,
              'ml_groups':{k:len(v) for k,v in ml.items()},'replay_rows':{k:len(v) for k,v in replay.items()},
              'overlength_replay_excluded':discarded}
    atomic_json(args.output/'manifest.json',manifest)
    initial=evaluate(model,tok,ml['dev'],encoded['dev'],replay['dev'])
    atomic_json(args.output/'baseline.json',initial)
    print('BASELINE',json.dumps({k:v for k,v in initial.items() if k!='results'}),flush=True)
    if args.epochs==0:return
    optimizer=torch.optim.AdamW([p for n,p in trainable],lr=args.lr,weight_decay=.01)
    micro=args.micro_groups;accum=args.accumulation
    batches=math.ceil(len(ml['train'])/micro)
    total_steps=math.ceil(batches/accum)*args.epochs
    warmup=max(1,round(total_steps*.03));step=0;history=[];started=time.monotonic()
    for epoch in range(1,args.epochs+1):
        model.train();order=list(range(len(ml['train'])));random.Random(args.seed+epoch).shuffle(order)
        replay_order=list(range(len(replay['train'])));random.Random(args.seed+1000+epoch).shuffle(replay_order)
        rp=0;optimizer.zero_grad(set_to_none=True);losses=[]
        for bi,start in enumerate(range(0,len(order),micro)):
            indices=order[start:start+micro]
            rows=[x for i in indices for x in encoded['train'][i]]
            inputs,labels=batch(tok,rows)
            logits=model(**inputs).logits.float()
            p,q=probabilities(logits)
            consistency=((p+q-1)**2).mean()
            divisor=min(accum,batches-(bi//accum)*accum)
            ml_loss=F.cross_entropy(logits,labels)
            ((.5*ml_loss+args.consistency*consistency)/divisor).backward()
            replay_rows=[replay['train'][replay_order[(rp+i)%len(replay_order)]] for i in range(len(rows))]
            rp+=len(rows)
            inputs,labels=batch(tok,replay_rows)
            replay_loss=F.cross_entropy(model(**inputs).logits.float(),labels)
            (.5*replay_loss/divisor).backward()
            losses.append([ml_loss.item(),replay_loss.item(),consistency.item()])
            if (bi+1)%accum==0 or bi+1==batches:
                step+=1
                factor=min(step/warmup,max(0.,(total_steps-step+1)/max(1,total_steps-warmup)))
                for group in optimizer.param_groups:group['lr']=args.lr*factor
                norm=torch.nn.utils.clip_grad_norm_([p for n,p in trainable],1.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient norm')
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                print(json.dumps({'epoch':epoch,'step':step,'total_steps':total_steps,'losses':losses[-1],'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
        directory=args.output/f'epoch_{epoch}'
        model.save_pretrained(directory);tok.save_pretrained(directory)
        metrics=evaluate(model,tok,ml['dev'],encoded['dev'],replay['dev'])
        metrics.update(epoch=epoch,mean_training_losses=np.mean(losses,axis=0).tolist())
        atomic_json(directory/'evaluation.json',metrics)
        history.append({k:v for k,v in metrics.items() if k!='results'})
        # Dev selection rule fixed before training: symmetric accuracy; retain
        # general NLI accuracy within two percentage points of the baseline.
        eligible=[m for m in history if m['replay_accuracy']>=initial['replay_accuracy']-.02]
        best=max(eligible,key=lambda m:(m['symmetric_accuracy'],m['swap_consistency'],-m['epoch'])) if eligible else None
        atomic_json(args.output/'summary.json',{'status':'complete' if epoch==args.epochs else 'running',
                    'history':history,'selected_epoch':best['epoch'] if best else None,
                    'selection':'dev symmetric accuracy, replay accuracy drop <= .02, tie: swap consistency then earlier epoch',
                    'elapsed_s':time.monotonic()-started})
        print('EPOCH',json.dumps(history[-1]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--consistency',type=float,default=.1)
    parser.add_argument('--lr',type=float,default=2e-5)
    parser.add_argument('--micro-groups',type=int,default=2)
    parser.add_argument('--accumulation',type=int,default=4)
    parser.add_argument('--max-length',type=int,default=2048)
    main(parser.parse_args())
