import os
import json
import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

app = FastAPI(title="OpenJev System One Server")

MODEL_DIR = os.getenv("MODEL_PATH", "/model/qwen3.5-4b-nli-v5")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] Loading OpenJev from {MODEL_DIR} on {DEVICE}...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    trust_remote_code=True
).to(DEVICE)
model.eval()
print("[✓] OpenJev model loaded successfully.")

TEMPLATE = "Premise: {premise}\nHypothesis: {hypothesis}"
RUBRIC_TEMPLATE = 'The answer to "{instr}" is {label}: {crit}'

MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "8192"))

ENT_INDEX = 1  # 0=contradiction, 1=entailment, 2=neutral

def predict_entailment_batch(pairs: list[tuple[str, str]]) -> list[float]:
    if not pairs:
        return []
    texts = [TEMPLATE.format(premise=p, hypothesis=h) for p, h in pairs]
    encoded = tok(texts, padding=False, truncation=False)
    lengths = [len(ids) for ids in encoded["input_ids"]]
    if max(lengths) > MAX_INPUT_TOKENS:
        raise HTTPException(status_code=413, detail=f"Input has {max(lengths)} tokens; limit {MAX_INPUT_TOKENS}. No truncation performed.")
    inputs = tok.pad(encoded, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits.float()
        probs = torch.softmax(logits, dim=-1)[:, ENT_INDEX]
    return probs.cpu().tolist()

@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL_DIR), "device": DEVICE, "max_input_tokens": MAX_INPUT_TOKENS, "truncation": False}

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

    return {"answers": answers}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
