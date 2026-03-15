"""Main baseline execution - Context assembly for code completion"""

import os
import random
import argparse
import jsonlines
from context_architect import ContextArchitect, FILE_SEP_SYMBOL, FILE_COMPOSE_FORMAT


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate context for code completion")

    # Context collection strategy
    parser.add_argument("--stage", type=str, default="dataset1", help="Stage of the project")
    parser.add_argument("--lang", type=str, default="python", help="Programming language")
    parser.add_argument("--strategy", type=str, default="architect", help="Context collection strategy")

    # Context trimming options
    parser.add_argument("--trim-prefix", action="store_true", help="Trim prefix to last 10 lines")
    parser.add_argument("--trim-suffix", action="store_true", help="Trim suffix to first 10 lines")

    # Token budgets
    parser.add_argument("--max-tokens", type=int, default=16000, help="Maximum tokens for context")
    parser.add_argument("--mellum-budget", type=int, default=8000, help="Token budget for Mellum (first priority)")

    return parser.parse_args()


def trim_prefix(prefix: str, lines: int = 10) -> str:
    prefix_lines = prefix.split("\n")
    if len(prefix_lines) > lines:
        prefix = "\n".join(prefix_lines[-lines:])
    return prefix


def trim_suffix(suffix: str, lines: int = 10) -> str:
    suffix_lines = suffix.split("\n")
    if len(suffix_lines) > lines:
        suffix = "\n".join(suffix_lines[:lines])
    return suffix


def assemble_random_context(root_directory: str) -> str:
    all_files = []
    for root, dirs, files in os.walk(root_directory):
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache'}]
        for file in files:
            if file.endswith('.py'):
                all_files.append(os.path.join(root, file))

    if not all_files:
        return ""

    file_path = random.choice(all_files)

    if not os.path.exists(file_path):
        return ""

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        file_content = f.read()

    clean_file_name = file_path[len(root_directory) + 1:]
    return FILE_COMPOSE_FORMAT.format(
        file_sep=FILE_SEP_SYMBOL,
        file_name=clean_file_name,
        file_content=file_content
    )


def main():
    args = parse_args()

    # Validation
    assert args.lang == "python", "Only Python is supported"
    print(f"Running the {args.strategy} baseline for stage '{args.stage}'")

    # Input/output paths
    completion_points_file = os.path.join("data", f"{args.lang}-{args.stage}.jsonl")

    prediction_file_name = f"{args.lang}-{args.stage}-{args.strategy}"
    if args.trim_prefix:
        prediction_file_name += "-short-prefix"
    if args.trim_suffix:
        prediction_file_name += "-short-suffix"
    predictions_file = os.path.join("predictions", f"{prediction_file_name}.jsonl")

    with jsonlines.open(completion_points_file, 'r') as reader:
        with jsonlines.open(predictions_file, 'w') as writer:
            for idx, datapoint in enumerate(reader):
                # Identify the repository storage for the datapoint
                repo_path = datapoint['repo'].replace("/", "__")
                repo_revision = datapoint['revision']
                root_directory = os.path.join(
                    "data",
                    f"repositories-{args.lang}-{args.stage}",
                    f"{repo_path}-{repo_revision}"
                )

                if not os.path.exists(root_directory):
                    print(f"[{idx}] Skipping {repo_path}: directory not found")
                    continue

                # Assemble context based on strategy
                if args.strategy == "architect":
                    architect = ContextArchitect(
                        root_directory,
                        args.max_tokens,
                        args.mellum_budget
                    )
                    context = architect.assemble_context(datapoint)
                else:
                    context = assemble_random_context(root_directory)

                submission = {"context": context}

                if args.trim_prefix:
                    submission["prefix"] = trim_prefix(datapoint["prefix"])
                if args.trim_suffix:
                    submission["suffix"] = trim_suffix(datapoint["suffix"])

                # Write result
                print(f"[{idx}] {datapoint['repo']}: {datapoint['path']} -> context")
                writer.write(submission)

    print(f"[DONE] Predictions written to {predictions_file}")


if __name__ == "__main__":
    main()
