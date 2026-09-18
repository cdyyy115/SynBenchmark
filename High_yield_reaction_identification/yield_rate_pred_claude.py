import argparse
import csv
import json
import os
import re
import time
from typing import List
import urllib.error
import urllib.request
from pathlib import Path

LIST_PREFIX_CHARS = "-*•"
LABEL_RE = re.compile(
    r"^\s*(?:final\s+answer|answer|prediction|predicted\s+answer|high-yielding\s+reaction)\s*[:：]\s*",
    flags=re.IGNORECASE,
)
NUMBER_RE = re.compile(r"^\s*\d+\s*[\.\)\:]\s*")
ANSWER_RE = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_candidate(text: str) -> str:
    text = str(text).strip()
    if not text:
        return ""

    inline = re.findall(r"`([^`]+)`", text)
    if inline:
        text = inline[0].strip()

    text = text.replace("```text", "").replace("```", "").strip()
    text = text.lstrip(LIST_PREFIX_CHARS).strip()
    text = NUMBER_RE.sub("", text)
    text = LABEL_RE.sub("", text)
    return text.strip().strip('"').strip("'").strip()


def normalize_answer(text: object) -> str:
    """Return canonical Yes/No if the text contains one clear yield label."""
    if text is None:
        return ""
    candidate = clean_candidate(str(text))
    if not candidate:
        return ""

    lowered = candidate.lower()
    if lowered in {"yes", "y", "true", "high", "high-yielding", "high yielding"}:
        return "Yes"
    if lowered in {"no", "n", "false", "low", "not high", "not high-yielding", "not high yielding"}:
        return "No"

    match = ANSWER_RE.search(candidate)
    if match:
        return "Yes" if match.group(1).lower() == "yes" else "No"
    return ""


def _json_candidates(text: str):
    text = text.strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
    except Exception:
        return []

    values = []
    if isinstance(obj, str):
        values.append(obj)
    elif isinstance(obj, list):
        values.extend(str(x) for x in obj if isinstance(x, (str, int, float, bool)))
    elif isinstance(obj, dict):
        for key in ("answer", "gold_answer", "prediction", "predicted_answer", "label", "yield_label", "high_yielding", "candidates"):
            value = obj.get(key)
            if isinstance(value, (str, int, float, bool)):
                values.append(str(value))
            elif isinstance(value, list):
                values.extend(str(x) for x in value if isinstance(x, (str, int, float, bool)))
    return values


def extract_candidates(text: object, max_candidates: int):
    """Extract Yes/No yield labels from model output."""
    if text is None:
        return []

    raw = str(text).strip()
    if not raw:
        return []

    pool = []
    pool.extend(_json_candidates(raw))

    for block in re.findall(r"```(?:text)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL):
        pool.extend(block.splitlines())

    pool.extend(raw.splitlines())
    pool.extend(re.findall(r"`([^`]+)`", raw))
    pool.append(raw)

    candidates = []
    seen = set()
    for item in pool:
        for piece in str(item).split("|||"):
            answer = normalize_answer(piece)
            if not answer or answer in seen:
                continue
            seen.add(answer)
            candidates.append(answer)
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def build_prompt(prompt: str, candidates_per_call: int) -> str:
    """Append a strict Yes/No output contract for automatic parsing."""
    n = max(1, candidates_per_call)
    if n == 1:
        output_rule = (
            "Return ONLY one word on a single line: Yes or No. "
            "Do not include explanations, labels, Markdown, numbering, or any other text."
        )
    else:
        output_rule = (
            f"Return ONLY {n} independent labels, one per line, and each line must be either Yes or No. "
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
    # Claude-compatible endpoint; thinking is disabled by omitting thinking/reasoning fields.
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
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



def load_existing_results(path: Path):
    """Load existing CSV rows for resume mode, keyed by (id, run_id)."""
    if not path.exists() or path.stat().st_size == 0:
        return {}

    existing = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = str(row.get("id", ""))
            try:
                run_id = int(row.get("run_id", 0))
            except (TypeError, ValueError):
                continue
            existing[(item_id, run_id)] = dict(row)
    return existing


def is_successful_existing_row(row):
    """A row is considered complete only when it has a prediction and no error."""
    if not row:
        return False
    pred = str(row.get("pred_yield_label", "")).strip()
    err = str(row.get("error", "")).strip()
    return bool(pred) and not err


def write_rows(path: Path, rows):
    fieldnames = [
        "id",
        "run_id",
        "class",
        "reaction_class",
        "uspto50k_class",
        "prompt_type",
        "num_candidates",
        "num_calls",
        "candidates_per_call",
        "reactants",
        "product",
        "yield_rate",
        "gold_answer",
        "reaction_smiles",
        "original_reaction_smiles",
        "raw_reagents",
        "raw_solvents",
        "raw_catalysts",
        "gold_reagents_smiles",
        "gold_solvents_smiles",
        "gold_catalysts_smiles",
        "pred_yield_label",
        "pred_yield_labels",
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
        description="Run reaction yield-rate prediction prompts with an OpenAI-compatible API, with Yes/No answer parsing."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "4.8-opus"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
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
        help="Total valid, unique Yes/No labels to collect for each prompt. Use 1 for this benchmark.",
    )
    parser.add_argument(
        "--candidates-per-call",
        type=int,
        default=1,
        help="How many Yes/No labels the prompt requests from each API call. Use 1 for this benchmark.",
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
        "--resume",
        action="store_true",
        help="Resume from an existing output CSV: skip successful (id, run_id) rows and retry failed/missing rows.",
    )
    parser.add_argument(
        "--max-calls-per-item",
        type=int,
        default=10,
        help="Safety cap to avoid infinite calls when a model repeatedly returns no parseable Yes/No answer.",
    )
    args = parser.parse_args()

    items = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        items = items[: args.limit]

    existing_by_key = load_existing_results(args.output_csv) if args.resume else {}
    results_by_key = dict(existing_by_key)

    if args.resume and existing_by_key:
        successful_count = sum(
            1 for row in existing_by_key.values() if is_successful_existing_row(row)
        )
        print(
            f"Resume mode: loaded {len(existing_by_key)} existing rows; "
            f"{successful_count} successful rows will be skipped."
        )

    for index, item in enumerate(items, start=1):
        item_id = str(item.get("id", ""))
        for run_id in range(1, args.num_runs + 1):
            key = (item_id, run_id)

            if args.resume and is_successful_existing_row(existing_by_key.get(key)):
                print(
                    f"[{index}/{len(items)} run={run_id}/{args.num_runs}] "
                    f"id={item.get('id')} SKIP (already successful)"
                )
                continue

            result = dict(item)
            result["run_id"] = run_id
            raw_outputs = []
            reasoning_outputs = []
            candidates = []

            try:
                call_count = 0
                while len(candidates) < args.num_candidates and call_count < args.max_calls_per_item:
                    call_count += 1
                    raw_output, reasoning_output = call_openai_chat(
                        prompt=build_prompt(item["prompt"], args.candidates_per_call),
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
                        "Model returned no parseable Yes/No yield label. "
                        f"Last content preview: {preview!r}"
                    )

                result["pred_yield_label"] = candidates[0]
                result["pred_yield_labels"] = " ||| ".join(candidates)
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
                result["pred_yield_label"] = candidates[0] if candidates else ""
                result["pred_yield_labels"] = " ||| ".join(candidates)
                result["num_candidates"] = len(candidates)
                result["num_calls"] = len(raw_outputs)
                result["candidates_per_call"] = args.candidates_per_call
                result["error"] = repr(exc)

            results_by_key[key] = result

            def _sort_key(k):
                item_key, run_key = k
                try:
                    input_pos = next(
                        i for i, x in enumerate(items)
                        if str(x.get("id", "")) == item_key
                    )
                except StopIteration:
                    input_pos = len(items)
                return (input_pos, int(run_key))

            ordered_rows = [
                results_by_key[k]
                for k in sorted(results_by_key.keys(), key=_sort_key)
            ]
            write_rows(args.output_csv, ordered_rows)

            print(
                f"[{index}/{len(items)} run={run_id}/{args.num_runs}] "
                f"id={item.get('id')} parsed={len(candidates)} calls={len(raw_outputs)} "
                f"error={bool(result['error'])}"
            )

    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
