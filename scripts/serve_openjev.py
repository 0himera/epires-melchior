import os
import json
import hashlib
from pathlib import Path
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

app = FastAPI(title="OpenJev System One Server")

MODEL_DIR = os.getenv("MODEL_PATH", "/model/qwen3.5-4b-nli-v5")
ADAPTER_DIR = os.getenv("ADAPTER_PATH")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] Loading OpenJev from {MODEL_DIR} on {DEVICE}...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
tok.padding_side = "right"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True, local_files_only=True, attn_implementation="sdpa"
).to(DEVICE)
for config in (model.config, model.config.get_text_config()):
    config.pad_token_id = tok.pad_token_id
    config.use_cache = False
MODEL_IDENTITY = {"base_model": os.path.basename(MODEL_DIR), "adapter": None,
                  "adapter_sha256": None}
if ADAPTER_DIR:
    from peft import PeftModel
    adapter_file = Path(ADAPTER_DIR) / "adapter_model.safetensors"
    MODEL_IDENTITY.update(adapter=os.getenv("ADAPTER_NAME", Path(ADAPTER_DIR).name),
                          adapter_sha256=hashlib.sha256(adapter_file.read_bytes()).hexdigest())
    model = PeftModel.from_pretrained(model, ADAPTER_DIR, local_files_only=True)
model.eval()
print(f"[✓] OpenJev model loaded: {MODEL_IDENTITY}", flush=True)

TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}"
RUBRIC_TEMPLATE = 'The answer to "{instr}" is {label}: {crit}'

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "8192"))
INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE", "1"))
if MAX_INPUT_TOKENS < 1 or INFERENCE_BATCH_SIZE < 1:
    raise ValueError("Input limit and inference batch size must be positive")

ENT_INDEX = 1  # 0=contradiction, 1=entailment, 2=neutral

def predict_entailment_batch(pairs: list[tuple[str, str]]) -> list[float]:
    if not pairs:
        return []
    texts = [TEMPLATE.format(premise=p, hypothesis=h) for p, h in pairs]
    encoded = tok(texts, padding=False, truncation=False)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    if max(lengths) > MAX_INPUT_TOKENS:
        raise HTTPException(status_code=413, detail=f"Input has {max(lengths)} tokens; limit {MAX_INPUT_TOKENS}. No truncation performed.")
    scores = []
    with torch.inference_mode():
        for start in range(0, len(pairs), INFERENCE_BATCH_SIZE):
            batch = {key: values[start:start + INFERENCE_BATCH_SIZE] for key, values in encoded.items()}
            inputs = tok.pad(batch, padding=True, return_tensors="pt").to(DEVICE)
            logits = model(**inputs).logits.float()
            scores.extend(torch.softmax(logits, dim=-1)[:, ENT_INDEX].cpu().tolist())
    return scores

@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_DIR), "model_identity": MODEL_IDENTITY,
            "device": DEVICE, "max_input_tokens": MAX_INPUT_TOKENS, "truncation": False,
            "inference_batch_size": INFERENCE_BATCH_SIZE}

@app.post("/v1/systemone")
@app.post("/api/v1/systemone")
async def systemone(request: Request):
    body = await request.json()
    state = body.get("state", "")
    if not isinstance(state, str):
        state = json.dumps(state, ensure_ascii=False)
    questions = body.get("questions", {})

    answers = {}

    for q_id, q_data in questions.items():
        q_type = q_data.get("type", "").lower()
        instr = q_data.get("instructions", "")

        if q_type == "noul":
            # Binary yes/no
            pairs = [
                (state, RUBRIC_TEMPLATE.format(instr=instr, label="yes", crit="affirmative / permitted / true")),
                (state, RUBRIC_TEMPLATE.format(instr=instr, label="no", crit="negative / prohibited / false"))
            ]
            ent_probs = predict_entailment_batch(pairs)
            p_yes = ent_probs[0]
            p_no = ent_probs[1]
            denom = max(p_yes + p_no, 1e-9)
            noul_prob = round(p_yes / denom, 4)
            answers[q_id] = {"noul": noul_prob}

        elif q_type == "choice":
            criteria = q_data.get("criteria", {})
            options = list(criteria.keys()) if criteria else ["yes", "no"]
            pairs = [
                (state, RUBRIC_TEMPLATE.format(instr=instr, label=opt, crit=criteria.get(opt, opt)))
                for opt in options
            ]
            ent_probs = predict_entailment_batch(pairs)
            denom = max(sum(ent_probs), 1e-9)
            norm_probs = {opt: round(p / denom, 4) for opt, p in zip(options, ent_probs)}
            best_opt = max(norm_probs.items(), key=lambda x: x[1])[0]
            answers[q_id] = {
                "choice": best_opt,
                "confidence": norm_probs[best_opt],
                "probabilities": norm_probs
            }

        elif q_type == "score":
            # Rating from 0.0 to 1.0 (graded from low to high)
            pairs = [
                (state, RUBRIC_TEMPLATE.format(instr=instr, label="high", crit="high quality, high performance, promising")),
                (state, RUBRIC_TEMPLATE.format(instr=instr, label="low", crit="low quality, poor performance, failing"))
            ]
            ent_probs = predict_entailment_batch(pairs)
            denom = max(ent_probs[0] + ent_probs[1], 1e-9)
            score_val = round(ent_probs[0] / denom, 4)
            answers[q_id] = {"score": score_val}

        else:
            answers[q_id] = {"error": f"Unknown question type: {q_type}"}

    return {"answers": answers, "model_identity": MODEL_IDENTITY}


@app.post("/v1/compare")
async def compare_candidates(request: Request):
    """Structured two-candidate comparison, invariant to display order.

    Returns original-orientation scores as diagnostics. General SystemOne calls
    retain their existing contract; arbitrary free-form states cannot be safely
    reordered without a structured candidate representation.
    """
    from melchior.crucible.decisions import comparison, nli_rows, symmetric_choice
    body = await request.json()
    try:
        rows = []
        for swapped in (False, True):
            payload = comparison(body['profile'], body['pair'], swapped=swapped,
                                 generation_swapped=body.get('generation_swapped', False))
            rows.extend(nli_rows(payload, 'A'))  # Labels unused during inference.
        scores = predict_entailment_batch([(r['premise'], r['hypothesis']) for r in rows])
        p = scores[0] / max(scores[0] + scores[1], 1e-12)
        q = scores[2] / max(scores[2] + scores[3], 1e-12)
        result = symmetric_choice(p, q)
        return {**result, 'normal_p_a': p, 'swapped_p_a': q,
                'model_identity': MODEL_IDENTITY,
                'score_semantics': 'mean relative entailment across both orientations'}
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
