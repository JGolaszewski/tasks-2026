#!/usr/bin/env python
"""Test script for architect strategy with a single repo."""

import os
import jsonlines
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import ContextArchitect

with jsonlines.open('../data/python-dataset1.jsonl', 'r') as reader:
    datapoint = reader.read()

print(f"Testing with datapoint:")
print(f"  Repo: {datapoint['repo']}")
print(f"  Path: {datapoint['path']}")
print(f"  Prefix length: {len(datapoint['prefix'])}")

repo_path = datapoint['repo'].replace("/", "__")
repo_revision = datapoint['revision']
root_directory = os.path.join("../data", "repositories-python-dataset1", f"{repo_path}-{repo_revision}")

print(f"  Root directory: {root_directory}")
print(f"  Exists: {os.path.exists(root_directory)}")

if os.path.exists(root_directory):
    print("\nInitializing ContextArchitect...")
    architect = ContextArchitect(root_directory, max_tokens=16000, mellum_budget=8000)

    print("Assembling context...")
    context = architect.assemble_context(datapoint)

    print(f"\nContext assembled!")
    print(f"  Context length: {len(context)} characters")
    print(f"  Context preview (first 500 chars):\n{context[:500]}")

    tokens = architect.count_tokens(context)
    print(f"  Estimated tokens: {tokens}")
else:
    print(f"ERROR: Root directory not found: {root_directory}")
