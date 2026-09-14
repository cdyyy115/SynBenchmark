import argparse
import csv
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

try:
    from rdkit import Chem
except ImportError:
    Chem = None


ATOM_MAP_PATTERN = re.compile(r":\d+(?=\])")
LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")

CHEMDFM_REACTION_PREDICTION_INSTRUCTION = (
    "Chemical reaction equations are typically expressed in the following form: "
    "reactant1.reactant2.reactant3...>reagent1.reagent2.reagent3...>product. "
    "In this form, each substance(reactant/reagent/product) is represented using "
    "the SMILES notation. Now we will provide you with an incomplete chemical "
    "reaction equation, where the missing part will be represented with \"\\___\\\". "
    "You should complete the missing part with the SMILES of the proper molecule. "
    "Based on the remaining portions of the reaction equation, please infer what "
    "the missing part could be. Please only provide the missing part in your "
    "response, without any additional content."
)


def strip_atom_mapping(smiles: str) -> str:
    return ATOM_MAP_PATTERN.sub("", smiles)


def resolve_reaction_column(df: pd.DataFrame, reaction_column: str | None) -> str:
    if reaction_column:
        if reaction_column not in df.columns:
            raise ValueError(
                f"Column {reaction_column!r} not found. Available: {list(df.columns)}"
            )
        return reaction_column

    for candidate in ("reactants>reagents>production", "reactants>reagents>product", "smiles"):
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not find a reaction SMILES column. Expected one of "
        "'reactants>reagents>production', 'reactants>reagents>product', or 'smiles'. "
        f"Available: {list(df.columns)}"
    )


def split_reaction_smiles(reaction_smiles: str) -> tuple[str, str, str]:
    parts = str(reaction_smiles).strip().split(">")
    if len(parts) != 3:
        raise ValueError(f"Expected reactants>reagents>product format: {reaction_smiles}")
    return parts[0], parts[1], parts[2]


def formatting_input(current_query: str, history: list[tuple[str, str]] | None = None) -> str:
    if history is None:
        history = []

    input_text = ""
    for idx, (query, answer) in enumerate(history):
        input_text += f"[Round {idx}]\n Human: {query}\n Assistant: {answer}\n"

    input_text += f"[Round {len(history)}]\n Human: {current_query}\n Assistant:"
    return input_text


def make_reaction_prediction_prompt(incomplete_equation: str) -> str:
    query = (
        CHEMDFM_REACTION_PREDICTION_INSTRUCTION
        + f"\n\nIncomplete equation: {incomplete_equation}\nCompletion:"
    )
    return formatting_input(query, history=[])


def make_incomplete_equation(
    reaction_smiles: str,
    keep_atom_mapping: bool,
) -> tuple[str, str, str, str]:
    reactants, reagents, product = split_reaction_smiles(reaction_smiles)
    if not keep_atom_mapping:
        reactants = strip_atom_mapping(reactants)
        reagents = strip_atom_mapping(reagents)
        product = strip_atom_mapping(product)

    incomplete_equation = f"{reactants}>{reagents}>\\___\\"
    known_left = f"{reactants}>{reagents}"
    return incomplete_equation, known_left, product, reagents


def clean_candidate(text: object) -> str:
    text = "" if pd.isna(text) else str(text).strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("smiles", "", 1).strip()
    first_line = text.splitlines()[0].strip()
    first_line = LIST_PREFIX_PATTERN.sub("", first_line).strip()
    return first_line.strip().strip('"').strip("'")


def extract_answer(decoded_text: str, prompt: str) -> str:
    if decoded_text.startswith(prompt):
        return decoded_text[len(prompt) :].strip()
    marker = "Assistant:"
    if marker in decoded_text:
        return decoded_text.rsplit(marker, 1)[-1].strip()
    return decoded_text.strip()


def canonicalize(smiles: object, keep_atom_mapping: bool) -> tuple[str, bool]:
    text = clean_candidate(smiles)
    if not keep_atom_mapping:
        text = strip_atom_mapping(text)
    if not text:
        return "", False
    if Chem is None:
        return text, True

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return "", False
    if not keep_atom_mapping:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True), True


def build_generation_config(args: argparse.Namespace, tokenizer) -> GenerationConfig:
    if args.official_demo_generation:
        return GenerationConfig(
            do_sample=True,
            top_k=20,
            top_p=0.9,
            temperature=0.9,
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generation_kwargs = {
        "do_sample": args.do_sample,
        "max_new_tokens": args.max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    return GenerationConfig(**generation_kwargs)


def write_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "id",
        "class",
        "reaction_smiles",
        "known_left",
        "incomplete_equation",
        "gold_product",
        "prompt",
        "pred_product",
        "pred_products",
        "raw_output",
        "gold_canonical",
        "pred_canonical",
        "pred_candidates_canonical",
        "valid_smiles",
        "top1_accuracy",
        "top3_accuracy",
        "top5_accuracy",
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
        description="Run ChemDFM forward reaction prediction on USPTO-50K CSV data."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/data/models/ChemDFM-v1.0-13B"))
    parser.add_argument("--reaction-column", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep-atom-mapping", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--official-demo-generation",
        action="store_true",
        help="Use the generation settings shown in the ChemDFM GitHub local inference demo.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    reaction_column = resolve_reaction_column(df, args.reaction_column)
    if args.limit is not None:
        df = df.head(args.limit)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    generation_config = build_generation_config(args, tokenizer)
    rows = []

    for row_index, row in tqdm(df.iterrows(), total=len(df), desc="ChemDFM forward prediction"):
        result = {
            "id": row.get("id", row_index),
            "class": row.get("class", ""),
            "reaction_smiles": row[reaction_column],
            "error": "",
        }
        try:
            incomplete_equation, known_left, gold_product, _ = make_incomplete_equation(
                row[reaction_column],
                keep_atom_mapping=args.keep_atom_mapping,
            )
            prompt = make_reaction_prediction_prompt(incomplete_equation)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    num_return_sequences=args.num_return_sequences,
                )

            raw_outputs = []
            candidates = []
            for output in outputs:
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                answer = extract_answer(decoded, prompt)
                raw_outputs.append(answer)
                candidate = clean_candidate(answer)
                if candidate:
                    candidates.append(candidate)

            candidates = candidates[: max(args.num_return_sequences, 1)]
            gold_canonical, gold_valid = canonicalize(
                gold_product,
                keep_atom_mapping=args.keep_atom_mapping,
            )
            canonical_candidates = [
                canonicalize(candidate, keep_atom_mapping=args.keep_atom_mapping)
                for candidate in candidates
            ]
            pred_canonicals = [canonical for canonical, _ in canonical_candidates]
            pred_validities = [valid for _, valid in canonical_candidates]

            result.update(
                {
                    "known_left": known_left,
                    "incomplete_equation": incomplete_equation,
                    "gold_product": gold_product,
                    "prompt": prompt,
                    "pred_product": candidates[0] if candidates else "",
                    "pred_products": " ||| ".join(candidates),
                    "raw_output": "\n---OUTPUT---\n".join(raw_outputs),
                    "gold_canonical": gold_canonical,
                    "pred_canonical": pred_canonicals[0] if pred_canonicals else "",
                    "pred_candidates_canonical": " ||| ".join(pred_canonicals),
                    "valid_smiles": bool(pred_validities[0]) if pred_validities else False,
                    "top1_accuracy": gold_valid and gold_canonical in pred_canonicals[:1],
                    "top3_accuracy": gold_valid and gold_canonical in pred_canonicals[:3],
                    "top5_accuracy": gold_valid and gold_canonical in pred_canonicals[:5],
                }
            )
        except Exception as exc:
            result["error"] = repr(exc)

        rows.append(result)
        write_rows(args.output_csv, rows)

    result_df = pd.DataFrame(rows)
    print(f"Saved predictions to: {args.output_csv}")
    if "valid_smiles" in result_df:
        print(f"Total samples: {len(result_df)}")
        print(f"Errors: {(result_df['error'].astype(str) != '').sum()}")
        print(f"Validity: {result_df['valid_smiles'].mean():.4f}")
        print(f"Top-1 accuracy: {result_df['top1_accuracy'].mean():.4f}")
        print(f"Top-3 accuracy: {result_df['top3_accuracy'].mean():.4f}")
        print(f"Top-5 accuracy: {result_df['top5_accuracy'].mean():.4f}")
    if Chem is None:
        print("WARNING: RDKit is not installed. Accuracy used cleaned string matching only.")


if __name__ == "__main__":
    main()
