import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from rdkit import Chem, RDLogger

# Avoid flooding stderr when testing non-SMILES text returned by an LLM.
RDLogger.DisableLog("rdApp.error")

LIST_PREFIX_CHARS = "-*•"
LABEL_RE = re.compile(
    r"^\s*(?:final\s+answer|answer|predicted\s+product(?:\s+smiles)?|product(?:\s+smiles)?|smiles)\s*[:：]\s*",
    flags=re.IGNORECASE,
)
NUMBER_RE = re.compile(r"^\s*\d+\s*[\.\)\:]\s*")


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_candidate(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""

    # If the model puts the SMILES inside inline Markdown code, prefer that text.
    inline = re.findall(r"`([^`]+)`", text)
    if inline:
        text = inline[0].strip()

    # Remove fenced-code markers and common labels/list prefixes.
    text = text.replace("```smiles", "").replace("```SMILES", "").replace("```", "").strip()
    text = text.lstrip(LIST_PREFIX_CHARS).strip()
    text = NUMBER_RE.sub("", text)
    text = LABEL_RE.sub("", text)

    # Remove surrounding quotes only; do not alter internal SMILES syntax.
    return text.strip().strip('"').strip("'").strip()


def is_valid_smiles(smiles: str) -> bool:
    """Return True only for a syntactically parseable, non-empty SMILES."""
    if not smiles or any(ch.isspace() for ch in smiles):
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return False
    return mol is not None and mol.GetNumAtoms() > 0


def _json_candidates(text: str):
    """Best-effort extraction when a model unexpectedly returns JSON."""
    text = text.strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except Exception:
        return []

    values: list[str] = []
    if isinstance(obj, str):
        values.append(obj)
    elif isinstance(obj, list):
        values.extend(str(x) for x in obj if isinstance(x, (str, int, float)))
    elif isinstance(obj, dict):
        for key in ("smiles", "product_smiles", "product", "answer", "candidates", "products"):
            value = obj.get(key)
            if isinstance(value, str):
                values.append(value)
            elif isinstance(value, list):
                values.extend(str(x) for x in value if isinstance(x, (str, int, float)))
    return values


def extract_candidates(text: object, max_candidates: int):
    """Extract only RDKit-valid product SMILES from model output."""
    if text is None:
        return []

    raw = str(text).strip()
    if not raw:
        return []

    pool: list[str] = []

    # 1) JSON-like outputs.
    pool.extend(_json_candidates(raw))

    # 2) Fenced code blocks.
    for block in re.findall(r"```(?:smiles)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL):
        pool.extend(block.splitlines())

    # 3) Normal line-by-line output.
    pool.extend(raw.splitlines())

    # 4) Inline backtick snippets.
    pool.extend(re.findall(r"`([^`]+)`", raw))

    candidates: list[str] = []
    seen: set[str] = set()

    for item in pool:
        # Support a common multi-answer delimiter without splitting '.' in SMILES.
        pieces = str(item).split("|||")
        for piece in pieces:
            candidate = clean_candidate(piece)
            if not is_valid_smiles(candidate):
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
            if len(candidates) >= max_candidates:
                return candidates

    return candidates


def build_prompt(prompt: str, candidates_per_call: int) -> str:
    """Append a model-agnostic output contract suitable for automatic SMILES parsing."""
    n = max(1, candidates_per_call)
    if n == 1:
        output_rule = (
            "Return ONLY the predicted major product SMILES on a single line. "
            "Do not include explanations, labels, Markdown, numbering, or any other text."
        )
    else:
        output_rule = (
            f"Return ONLY {n} predicted product SMILES, one SMILES per line. "
            "Do not include explanations, labels, Markdown, numbering, or any other text."
        )
    return f"{prompt.rstrip()}\n\nIMPORTANT OUTPUT FORMAT:\n{output_rule}"


def call_openai_chat(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
    candidates_per_call: int,
):
    # Do not send Qwen-specific chat_template_kwargs here.
    # GLM-5.3 performs thinking by default/force; final answer is expected in message.content.
    payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": temperature,
    "top_p": top_p,
    "max_tokens": max_tokens,
    "thinking": {
        "type": "disabled"
    }
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
    reasoning_content = message.get("reasoning_content")

    # Final answer should come from content. reasoning_content is saved only for debugging.
    content_text = "" if content is None else str(content)
    reasoning_text = "" if reasoning_content is None else str(reasoning_content)
    return content_text, reasoning_text


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
        "reasoning_output",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run forward reaction prediction prompts with an OpenAI-compatible API, with RDKit SMILES validation."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "glm-5.3"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.z.ai/api/paas/v4"),
        help="OpenAI-compatible base URL.",
    )
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", "OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=1,
        help="Total valid, unique SMILES candidates to collect for each prompt.",
    )
    parser.add_argument(
        "--candidates-per-call",
        type=int,
        default=1,
        help="How many product SMILES the prompt requests from each API call.",
    )
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=400)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Independent repeated runs for each prompt.",
    )
    parser.add_argument(
        "--max-calls-per-item",
        type=int,
        default=10,
        help="Safety cap to avoid infinite calls when a model repeatedly returns no valid SMILES.",
    )
    args = parser.parse_args()

    items = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        items = items[: args.limit]

    results = []
    for index, item in enumerate(items, start=1):
        for run_id in range(1, args.num_runs + 1):
            result = dict(item)
            result["run_id"] = run_id
            raw_outputs: list[str] = []
            reasoning_outputs: list[str] = []
            candidates: list[str] = []

            try:
                call_count = 0
                while len(candidates) < args.num_candidates and call_count < args.max_calls_per_item:
                    call_count += 1
                    raw_output, reasoning_output = call_openai_chat(
                        prompt=item["prompt"],
                        model=args.model,
                        api_key=args.api_key,
                        base_url=args.base_url,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        candidates_per_call=args.candidates_per_call,
                    )
                    raw_outputs.append(raw_output)
                    reasoning_outputs.append(reasoning_output)

                    extracted = extract_candidates(raw_output, args.candidates_per_call)
                    for candidate in extracted:
                        if candidate not in candidates:
                            candidates.append(candidate)
                            if len(candidates) >= args.num_candidates:
                                break

                    if not raw_output:
                        break
                    if len(candidates) < args.num_candidates:
                        time.sleep(args.sleep)

                candidates = candidates[: args.num_candidates]
                result["raw_output"] = "\n---CALL---\n".join(raw_outputs)
                result["reasoning_output"] = "\n---CALL---\n".join(reasoning_outputs)

                if not candidates:
                    preview = raw_outputs[-1][:500].replace("\n", " ") if raw_outputs else "<empty>"
                    raise ValueError(
                        "Model returned no RDKit-valid product SMILES candidates. "
                        f"Last content preview: {preview!r}"
                    )

                result["pred_product"] = candidates[0]
                result["pred_products"] = " ||| ".join(candidates)
                result["num_candidates"] = len(candidates)
                result["num_calls"] = len(raw_outputs)
                result["candidates_per_call"] = args.candidates_per_call
                result["error"] = ""

            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                KeyError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                # Keep raw outputs instead of erasing them; this makes API/model-format issues debuggable.
                result["raw_output"] = "\n---CALL---\n".join(raw_outputs)
                result["reasoning_output"] = "\n---CALL---\n".join(reasoning_outputs)
                result["pred_product"] = candidates[0] if candidates else ""
                result["pred_products"] = " ||| ".join(candidates)
                result["num_candidates"] = len(candidates)
                result["num_calls"] = len(raw_outputs)
                result["candidates_per_call"] = args.candidates_per_call
                result["error"] = repr(exc)

            results.append(result)
            write_rows(args.output_csv, results)
            print(
                f"[{index}/{len(items)} run={run_id}/{args.num_runs}] "
                f"id={item.get('id')} valid={len(candidates)} calls={len(raw_outputs)} "
                f"error={bool(result['error'])}"
            )

    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
