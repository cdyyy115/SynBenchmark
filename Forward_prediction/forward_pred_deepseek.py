import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from rdkit import Chem

LIST_PREFIX_CHARS = "-*•"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_candidate(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("smiles", "", 1).strip()
    text = text.strip()
    text = text.lstrip(LIST_PREFIX_CHARS).strip()
    if "." in text:
        prefix, rest = text.split(".", 1)
        if prefix.strip().isdigit():
            text = rest.strip()
    return text.strip().strip('"').strip("'")

def is_valid_smiles(text):
    if not text or any(ch.isspace() for ch in text.strip()):
        return False
    return Chem.MolFromSmiles(text.strip()) is not None

def extract_candidates(text, max_candidates):
    if text is None:
        return []
    text = str(text)

    candidates = []
    for line in text.strip().splitlines():
        candidate = clean_candidate(line)
        if is_valid_smiles(candidate):
            candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break

    return candidates


def call_openai_chat(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
    thinking: str,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "thinking": {"type": thinking},
    }
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    message = data["choices"][0].get("message", {})
    content = message.get("content")
    return "" if content is None else str(content)


def write_rows(path: Path, rows):
    fieldnames = [
        "id",
        "run_id",
        "class",
        "prompt_type",
        "num_candidates",
        "num_calls",
        "candidates_per_call",
        "reactants",
        "gold_product",
        "pred_product",
        "pred_products",
        "raw_output",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run forward reaction prediction prompts with an OpenAI-compatible API."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument("--api_key", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default="disabled",
        help="DeepSeek thinking mode. Default: disabled for direct prediction benchmarks.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="Total candidates to collect for each prompt.",
    )
    parser.add_argument(
        "--candidates-per-call",
        type=int,
        default=1,
        help="Candidates requested from each API call. Use 1 for repeated sampling.",
    )
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Independent repeated calls for each prompt.",
    )
    args = parser.parse_args()

    # api_key = os.getenv(args.api_key_env)
    # if not api_key:
    #     raise SystemExit(f"Missing API key environment variable: {args.api_key_env}")

    items = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        items = items[: args.limit]

    results = []
    for index, item in enumerate(items, start=1):
        for run_id in range(1, args.num_runs + 1):
            result = dict(item)
            result["run_id"] = run_id
            try:
                raw_outputs = []
                candidates = []
                while len(candidates) < args.num_candidates:
                    raw_output = call_openai_chat(
                        prompt=item["prompt"],
                        model=args.model,
                        api_key=args.api_key,
                        base_url=args.base_url,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        thinking=args.thinking,
                    )
                    raw_outputs.append(raw_output)
                    candidates.extend(extract_candidates(raw_output, args.candidates_per_call))
                    time.sleep(args.sleep)

                    if not raw_output:
                        break

                candidates = candidates[: args.num_candidates]
                result["raw_output"] = "\n---CALL---\n".join(raw_outputs)
                if not candidates:
                    raise ValueError("Model returned no product SMILES candidates.")
                result["pred_product"] = candidates[0] if candidates else ""
                result["pred_products"] = " ||| ".join(candidates)
                result["num_candidates"] = args.num_candidates
                result["num_calls"] = len(raw_outputs)
                result["candidates_per_call"] = args.candidates_per_call
                result["error"] = ""
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                KeyError,
                ValueError,
            ) as exc:
                result["raw_output"] = ""
                result["pred_product"] = ""
                result["pred_products"] = ""
                result["num_candidates"] = args.num_candidates
                result["num_calls"] = 0
                result["candidates_per_call"] = args.candidates_per_call
                result["error"] = repr(exc)

            results.append(result)
            write_rows(args.output_csv, results)
            print(
                f"[{index}/{len(items)} run={run_id}/{args.num_runs}] "
                f"id={item.get('id')} error={bool(result['error'])}"
            )

    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
