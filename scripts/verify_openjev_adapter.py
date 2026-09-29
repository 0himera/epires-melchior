"""Reload a saved adapter and compare its dev predictions to the training run."""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from scripts.train_openjev_adapter import read_rows, encode_rows, batch, probabilities


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    summary=json.loads((args.run/'summary.json').read_text())
    epoch=summary['selected_epoch']
    if epoch is None:
        (args.run/'reload_verification.json').write_text(json.dumps({'verified':False,'reason':'No epoch passed the retention gate'}))
        raise SystemExit(0)
    adapter=args.run/f'epoch_{epoch}'
    reference=json.loads((adapter/'evaluation.json').read_text())['results'][:4]
    groups=read_rows(args.data/'ml_dev.jsonl')[:4]
    tok=AutoTokenizer.from_pretrained(adapter,local_files_only=True)
    base=AutoModelForSequenceClassification.from_pretrained(args.model,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    base.config.pad_token_id=tok.pad_token_id
    base.config.get_text_config().pad_token_id=tok.pad_token_id
    base.config.use_cache=False;base.config.get_text_config().use_cache=False
    model=PeftModel.from_pretrained(base,adapter,local_files_only=True).eval()
    results=[]
    with torch.inference_mode():
        for start in range(0,len(groups),2):
            rows=[r for g in groups[start:start+2] for r in g['views']]
            enc,_=encode_rows(tok,rows,2048)
            inputs,_=batch(tok,enc)
            a,b=probabilities(model(**inputs).logits)
            results.extend(zip(a.cpu().tolist(),b.cpu().tolist()))
    error=max(abs(got[k]-ref[field]) for got,ref in zip(results,reference) for k,field in enumerate(('p_a','reverse_p_a')))
    report={'verified':error<1e-4,'epoch':epoch,'pairs':len(results),'max_score_error':error}
    (args.run/'reload_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
    if not report['verified']:raise SystemExit('Reloaded adapter does not reproduce saved predictions')
