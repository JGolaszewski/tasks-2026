#!/usr/bin/env python
"""
Verification script for the three-phase context assembly system.
Tests both strategies and validates output format.
"""

import os
import sys
import jsonlines


def verify_file_exists(filepath):
    if not os.path.exists(filepath):
        print(f" File not found: {filepath}")
        return False
    print(f"✓ File found: {filepath}")
    return True


def verify_baselines_py():
    """Verify baselines.py has all required components."""
    with open('../baselines.py', 'r') as f:
        content = f.read()

    required_components = {
        'KnowledgeGraphBuilder': 'Phase I: Static Analysis',
        'clean_code': 'Phase II: Deterministic Filtering',
        'ContextArchitect': 'Phase III: Heuristic Ranking',
        'assemble_context': 'Context Assembly Method',
        'FILE_SEP_SYMBOL': 'File Separator Token',
        'mellum_budget': 'Mellum Token Budget',
    }

    print("\n📋 Verifying baselines.py components:")
    all_present = True
    for component, description in required_components.items():
        if component in content:
            print(f"✓ {component:30} ({description})")
        else:
            print(f" {component:30} - MISSING!")
            all_present = False

    return all_present


def verify_output_format():
    """Verify output JSONL format."""
    print("\n Verifying output format:")

    files_to_check = [
        'predictions/python-dataset1-random.jsonl',
        'predictions/python-dataset1-architect.jsonl',
    ]

    for filepath in files_to_check:
        if not os.path.exists(filepath):
            print(f"⚠ File not found (may need to run baselines first): {filepath}")
            continue

        try:
            with jsonlines.open(filepath, 'r') as reader:
                entry = reader.read()
                required_keys = {'context'}
                if all(k in entry for k in required_keys):
                    context_len = len(entry['context'])
                    print(f"✓ {filepath}")
                    print(f"  - Context length: {context_len} characters")
                    print(f"  - Keys: {list(entry.keys())}")
                else:
                    print(f" {filepath} - Missing required keys")
                    return False
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return False

    return True


def verify_strategies():
    """Verify both strategies are implemented."""
    print("\n🎯 Verifying strategies:")

    with open('../baselines.py', 'r') as f:
        content = f.read()

    strategies = {
        'architect': 'ContextArchitect',
        'random': 'random.choice',
    }

    all_present = True
    for strategy, indicator in strategies.items():
        if f'strategy == "{strategy}"' in content or (strategy == 'random' and indicator in content):
            print(f"✓ {strategy:15} strategy implemented")
        else:
            print(f"{strategy:15} strategy missing")
            all_present = False

    return all_present


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("  EnsembleAI2026 Context Assembly System - Verification Suite")
    print("=" * 70)

    checks = [
        ("baselines.py exists", lambda: verify_file_exists('../baselines.py')),
        ("baselines.py components", verify_baselines_py),
        ("strategies implemented", verify_strategies),
        ("output format", verify_output_format),
    ]

    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n⚠ Error during {check_name}: {e}")
            results.append((check_name, False))

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for check_name, result in results:
        status = "✓ PASS" if result else "FAIL"
        print(f"{status:10} {check_name}")

    print(f"\nTotal: {passed}/{total} checks passed")

    if passed == total:
        print("\n All verification checks passed!")
        print("\nYou can now run:")
        print("  python baselines.py --strategy architect --stage dataset1")
        print("  python baselines.py --strategy random --stage dataset1")
        return 0
    else:
        print(f"\n⚠ {total - passed} check(s) failed. Please review the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
