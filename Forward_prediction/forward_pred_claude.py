import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path


LIST_PREFIX_CHARS = "-*•"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_candidate(text: str) -> str:
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


def extract_candidates(text: object, max_candidates: int):
    if text is None:
        return []

    text = str(text)
    candidates = []

    for line in text.strip().splitlines():
        candidate = clean_candidate(line)
        if candidate:
            candidates.append(candidate)
        if len(candidates) >= max_candidates:
            break

    if not candidates:
        cleaned = clean_candidate(text)
        if cleaned:
            candidates.append(cleaned)

    return candidates


def call_openai_chat(
    prompt: str,
    model: str,
    api_key: str,
    base_url: str,
    max_tokens: int,
    timeout: int,
) -> str:
    """
    Claude Opus 4.8 through an OpenAI-compatible /chat/completions endpoint.

    Thinking is disabled by simply not sending any thinking/reasoning parameter.
    Temperature/top_p are also omitted intentionally.
    """
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

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_existing_results(path: Path):
    """
    Read an existing output CSV for resume mode.

    A run is considered completed only when:
      1) pred_product is non-empty
      2) error is empty

    Therefore old 402/429/503/timeout/empty-output rows will be retried.
    """
    if not path.exists():
        return [], set()

    rows = []
    completed = set()

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

            item_id = str(row.get("id", ""))
            run_id = str(row.get("run_id", ""))
            pred_product = str(row.get("pred_product", "")).strip()
            error = str(row.get("error", "")).strip()

            if pred_product and not error:
                completed.add((item_id, run_id))

    return rows, completed


def replace_result(results, new_result):
    """
    Ensure each (id, run_id) appears only once.
    Old failed rows are replaced by the newest retry result.
    """
    new_id = str(new_result.get("id", ""))
    new_run_id = str(new_result.get("run_id", ""))

    filtered = [
        row
        for row in results
        if not (
            str(row.get("id", "")) == new_id
            and str(row.get("run_id", "")) == new_run_id
        )
    ]
    filtered.append(new_result)
    return filtered


def request_with_retry(
    prompt,
    model,
    api_key,
    base_url,
    max_tokens,
    timeout,
    max_retries,
    retry_base_sleep,
):
    """
    Retry transient HTTP/network failures.

    HTTP 402:
        Raised immediately; caller stops the whole benchmark.

    HTTP 429/500/502/503/504:
        Exponential backoff retry.

    Other HTTP errors:
        Raised immediately.
    """
    retryable_http_codes = {429, 500, 502, 503, 504}

    for attempt in range(max_retries + 1):
        try:
            return call_openai_chat(
                prompt=prompt,
                model=model,
                api_key=api_key,
                base_url=base_url,
                max_tokens=max_tokens,
                timeout=timeout,
            )

        except urllib.error.HTTPError as exc:
            if exc.code == 402:
                raise

            if exc.code not in retryable_http_codes:
                raise

            if attempt >= max_retries:
                raise

            wait_seconds = retry_base_sleep * (2 ** attempt)
            print(
                f"  HTTP {exc.code}; retry "
                f"{attempt + 1}/{max_retries} after {wait_seconds:.1f}s"
            )
            time.sleep(wait_seconds)

        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= max_retries:
                raise

            wait_seconds = retry_base_sleep * (2 ** attempt)
            print(
                f"  {type(exc).__name__}; retry "
                f"{attempt + 1}/{max_retries} after {wait_seconds:.1f}s"
            )
            time.sleep(wait_seconds)

    raise RuntimeError("Retry loop ended unexpectedly.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run forward reaction prediction with Claude Opus 4.8 "
            "through an OpenAI-compatible API, with resume and retry support."
        )
    )

    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)

    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "claude-opus-4-8"),
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible base URL.",
    )

    # Keeps compatibility with your original usage:
    # --api_key can directly receive the API key string.
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""))

    parser.add_argument("--max-tokens", type=int, default=1024)
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
        help="Candidates extracted from each API call.",
    )
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Independent repeated calls for each prompt.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries for HTTP 429/500/502/503/504 or network errors.",
    )
    parser.add_argument(
        "--retry-base-sleep",
        type=float,
        default=5.0,
        help="Initial sleep seconds for exponential backoff retries.",
    )

    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit(
            "Missing API key. Pass --api_key YOUR_KEY "
            "or set OPENAI_API_KEY."
        )

    items = load_jsonl(args.input_jsonl)
    if args.limit is not None:
        items = items[: args.limit]

    # Resume from previous output CSV if it exists.
    results, completed = load_existing_results(args.output_csv)

    if results:
        print(
            f"Resume mode: found {len(results)} existing rows; "
            f"{len(completed)} successful runs will be skipped."
        )
    else:
        print("No existing output CSV found. Starting from the beginning.")

    for index, item in enumerate(items, start=1):
        for run_id in range(1, args.num_runs + 1):
            item_id = str(item.get("id", ""))
            key = (item_id, str(run_id))

            if key in completed:
                print(
                    f"[{index}/{len(items)} run={run_id}/{args.num_runs}] "
                    f"id={item.get('id')} SKIP (already completed)"
                )
                continue

            result = dict(item)
            result["run_id"] = run_id

            try:
                raw_outputs = []
                candidates = []

                while len(candidates) < args.num_candidates:
                    raw_output = request_with_retry(
                        prompt=item["prompt"],
                        model=args.model,
                        api_key=args.api_key,
                        base_url=args.base_url,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        max_retries=args.max_retries,
                        retry_base_sleep=args.retry_base_sleep,
                    )

                    raw_outputs.append(raw_output)
                    candidates.extend(
                        extract_candidates(
                            raw_output,
                            args.candidates_per_call,
                        )
                    )

                    time.sleep(args.sleep)

                    if not raw_output:
                        break

                candidates = candidates[: args.num_candidates]

                result["raw_output"] = "\n---CALL---\n".join(raw_outputs)

                if not candidates:
                    raise ValueError(
                        "Model returned no product SMILES candidates."
                    )

                result["pred_product"] = candidates[0]
                result["pred_products"] = " ||| ".join(candidates)
                result["num_candidates"] = args.num_candidates
                result["num_calls"] = len(raw_outputs)
                result["candidates_per_call"] = args.candidates_per_call
                result["error"] = ""

                results = replace_result(results, result)
                write_rows(args.output_csv, results)

                completed.add(key)

                print(
                    f"[{index}/{len(items)} run={run_id}/{args.num_runs}] "
                    f"id={item.get('id')} OK"
                )

            except urllib.error.HTTPError as exc:
                result["raw_output"] = ""
                result["pred_product"] = ""
                result["pred_products"] = ""
                result["num_candidates"] = args.num_candidates
                result["num_calls"] = 0
                result["candidates_per_call"] = args.candidates_per_call
                result["error"] = repr(exc)

                results = replace_result(results, result)
                write_rows(args.output_csv, results)

                if exc.code == 402:
                    raise SystemExit(
                        "\nHTTP 402 Payment Required.\n"
                        "The API balance/quota is likely exhausted.\n"
                        "Recharge the account or replace the API key, "
                        "then rerun EXACTLY the same command.\n"
                        "Successful (id, run_id) rows will be skipped, "
                        "and this failed row will be retried."
                    )

                print(
                    f"[{index}/{len(items)} run={run_id}/{args.num_runs}] "
                    f"id={item.get('id')} SKIP "
                    f"(HTTP {exc.code}: {exc.reason})"
                )

            except (
                urllib.error.URLError,
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

                results = replace_result(results, result)
                write_rows(args.output_csv, results)

                print(
                    f"[{index}/{len(items)} run={run_id}/{args.num_runs}] "
                    f"id={item.get('id')} SKIP ({repr(exc)})"
                )

    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
