from datasets import load_dataset
import json
ds = load_dataset("databricks/databricks-dolly-15k")["train"].shuffle(seed=42)
split = int(0.9 * len(ds))
train, eval = ds.select(range(split)), ds.select(range(split, len(ds)))
def dump(d, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in d:
            rec = {
                "instruction": ex["instruction"].strip(),
                "input": (ex.get("context") or "").strip(),
                "output": ex["response"].strip(),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
dump(train, "train.jsonl"); dump(eval, "eval.jsonl")
print("wrote train.jsonl and eval.jsonl")