import os
import io
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except Exception:
    VLLM_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Cache teacher outputs")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--teacher", required=True, help="Teacher model")
    parser.add_argument("--output", required=True, help="Path to LMDB output")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--mode", choices=["logits", "hidden"], default="logits")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM if available")
    return parser.parse_args()


def sharegpt_format(example, tokenizer):
    conversations = example["conversations"]
    message = []
    if isinstance(conversations, list):
        for conv in conversations:
            if conv.get("from") == "human":
                message.append({"role": "user", "content": conv.get("value", "")})
            elif conv.get("from") == "gpt":
                message.append({"role": "assistant", "content": conv.get("value", "")})
            elif conv.get("from") == "system":
                message.insert(0, {"role": "system", "content": conv.get("value", "")})
    if not any(m.get("role") == "system" for m in message):
        message.insert(0, {"role": "system", "content": "You are a helpful assistant."})
    text = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return {"text": text}


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset, split=args.split)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher)

    dataset = dataset.map(lambda ex: sharegpt_format(ex, tokenizer), remove_columns=dataset.column_names)

    def tok_fn(examples, indices):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tokens["id"] = indices
        return tokens

    dataset = dataset.map(tok_fn, batched=True, with_indices=True)

    import lmdb
    env = lmdb.open(args.output, map_size=int(1e12))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.use_vllm and VLLM_AVAILABLE and args.mode == "logits":
        llm = LLM(model=args.teacher)
        for ex in dataset:
            prompt = tokenizer.decode(ex["input_ids"])
            outputs = llm.generate([prompt])[0]
            logits = torch.tensor(outputs.logits, dtype=torch.float16)
            buf = io.BytesIO()
            torch.save(logits, buf)
            with env.begin(write=True) as txn:
                txn.put(str(ex["id"]).encode(), buf.getvalue())
    else:
        model = AutoModelForCausalLM.from_pretrained(args.teacher).to(device)
        model.eval()
        for ex in dataset:
            input_ids = torch.tensor([ex["input_ids"]]).to(device)
            attention_mask = torch.tensor([ex["attention_mask"]]).to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=args.mode=="hidden")
            data = out.logits[0].cpu() if args.mode=="logits" else [h[0].cpu() for h in out.hidden_states]
            buf = io.BytesIO()
            torch.save(data, buf)
            with env.begin(write=True) as txn:
                txn.put(str(ex["id"]).encode(), buf.getvalue())


if __name__ == "__main__":
    main()
