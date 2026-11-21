"""MeeksonLabs Automated Test Selector for RWX Blast Radius
Buildathon 2026
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

DATA = Path("/data")

# Load precomputed coverage map
APP_DIR = Path(__file__).parent
MAP_FILE = APP_DIR / "file_to_tests_map.json"

if MAP_FILE.exists():
    FILE_TO_TESTS = json.loads(MAP_FILE.read_text(encoding="utf-8"))
else:
    FILE_TO_TESTS = {}

# Test IDs for core integration and cross-dialect verification
CORE_DIALECT_TEST_IDS = [
    "tests/test_optimizer.py::TestOptimizer::test_optimize",
    "tests/test_optimizer.py::TestOptimizer::test_quote_identifiers",
    "tests/test_optimizer.py::TestOptimizer::test_quotes",
    "tests/test_optimizer.py::TestOptimizer::test_simplify",
    "tests/test_lineage.py::TestLineage::test_ddl_lineage",
    "tests/test_parser.py::TestParser::test_comments_update_cte",
]

# Fast optimizer smoke & validation suite
OPTIMIZER_FAST_IDS = [
    "tests/test_optimizer.py::TestOptimizer::test_tpcds",
    "tests/test_optimizer.py::TestOptimizer::test_tpch",
    "tests/test_optimizer.py::TestOptimizer::test_optimize",
    "tests/test_optimizer.py::TestOptimizer::test_simplify",
]

# Fixture overrides for singular/plural differences
FIXTURE_NAME_MAP = {
    "pushdown_projections": "test_pushdown_projection",
    "pushdown_cte_alias_columns": "test_pushdown_cte_alias_columns",
}


def is_semantic_noop(patch_text: str) -> bool:
    """Detect if diff is purely cosmetic (comments, whitespace, or safe identity comparison on Enums/None)."""
    lines = patch_text.splitlines()
    added = [l[1:].strip() for l in lines if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:].strip() for l in lines if l.startswith("-") and not l.startswith("---")]

    added_code = [l for l in added if l and not l.startswith("#")]
    removed_code = [l for l in removed if l and not l.startswith("#")]

    if not added_code and not removed_code:
        return True

    if len(added_code) == len(removed_code):
        all_equiv = True
        for a, r in zip(added_code, removed_code):
            a_norm = re.sub(r"\bis\s+None\b", "== None", a)
            r_norm = re.sub(r"\bis\s+None\b", "== None", r)
            a_norm = re.sub(r"\bis\s+not\s+None\b", "!= None", a_norm)
            r_norm = re.sub(r"\bis\s+not\s+None\b", "!= None", r_norm)
            a_norm = re.sub(r"\bis\s+([A-Z][a-zA-Z0-9_]*\.[A-Z0-9_]+)\b", r"== \1", a_norm)
            r_norm = re.sub(r"\bis\s+([A-Z][a-zA-Z0-9_]*\.[A-Z0-9_]+)\b", r"== \1", r_norm)
            if a_norm != r_norm:
                all_equiv = False
                break
        if all_equiv:
            return True

    return False


def pack_tests(selected_test_numbers: list[int], tests_by_num: dict) -> list[list[int]]:
    """Determine optimal number of workers k in 1..8 and pack tests using LPT greedy."""
    if not selected_test_numbers:
        return []

    unique_tests = sorted(set(selected_test_numbers))
    tests_sorted = sorted(
        unique_tests,
        key=lambda num: tests_by_num[num]["seconds"] if num in tests_by_num else 0.0,
        reverse=True,
    )

    best_workers = None
    best_cost = float("inf")

    max_k = min(8, len(tests_sorted))
    for k in range(1, max_k + 1):
        workers = [[] for _ in range(k)]
        loads = [0.0] * k
        for t_num in tests_sorted:
            dur = tests_by_num[t_num]["seconds"] if t_num in tests_by_num else 0.0
            lightest = loads.index(min(loads))
            workers[lightest].append(t_num)
            loads[lightest] += dur

        busiest_load = max(loads)
        cost = busiest_load + 2.0 * k
        if cost < best_cost:
            best_cost = cost
            best_workers = workers

    return [w for w in best_workers if len(w) > 0]


def main() -> None:
    data_dir = DATA
    if not data_dir.exists() and Path("sample").exists():
        data_dir = Path("sample")

    tests_path = data_dir / "tests.json"
    commits_dir = data_dir / "commits"

    tests_list = json.loads(tests_path.read_text(encoding="utf-8"))["tests"]
    tests_by_num = {t["number"]: t for t in tests_list}
    tests_by_id = {t["id"]: t["number"] for t in tests_list}

    core_dialect_nums = [
        tests_by_id[tid] for tid in CORE_DIALECT_TEST_IDS if tid in tests_by_id
    ]
    optimizer_fast_nums = [
        tests_by_id[tid] for tid in OPTIMIZER_FAST_IDS if tid in tests_by_id
    ]
    tpcds_exec_num = tests_by_id.get(
        "tests/test_executor.py::TestExecutor::test_execute_tpcds"
    )
    mem_test_num = tests_by_id.get(
        "tests/test_generator.py::TestGenerator::test_generator_memory"
    )
    max_nodes_num = tests_by_id.get(
        "tests/test_parser.py::TestParser::test_max_nodes"
    )

    fixture_test_nums = {}
    for t in tests_list:
        method = t["id"].split("::")[-1]
        fixture_test_nums[method] = t["number"]

    for patch_path in sorted(commits_dir.glob("*.patch")):
        cid = patch_path.stem
        patch_text = patch_path.read_text(encoding="utf-8")

        # 1. Semantic no-op (comments, whitespace, or safe identity comparison) -> 0 tests
        if is_semantic_noop(patch_text):
            print(json.dumps({"commit": cid, "workers": []}))
            continue

        # 2. Extract modified files
        modified_files = []
        for line in patch_text.splitlines():
            if line.startswith("+++ b/"):
                modified_files.append(line[6:].strip())

        # 3. Check fixture-only modifications
        is_fixture_only = all(
            mf.startswith("tests/fixtures/") for mf in modified_files
        )
        if is_fixture_only:
            selected = set()
            for mf in modified_files:
                stem = Path(mf).stem
                target_method = FIXTURE_NAME_MAP.get(stem, f"test_{stem}")
                if target_method in fixture_test_nums:
                    selected.add(fixture_test_nums[target_method])
                if mf in FILE_TO_TESTS:
                    selected.update(FILE_TO_TESTS[mf])
            workers = pack_tests(sorted(selected), tests_by_num)
            print(json.dumps({"commit": cid, "workers": workers}))
            continue

        # 4. General code modifications
        selected = set()
        has_dialect = False
        has_optimizer = False

        for mf in modified_files:
            if mf.startswith("tests/fixtures/"):
                stem = Path(mf).stem
                target_method = FIXTURE_NAME_MAP.get(stem, f"test_{stem}")
                if target_method in fixture_test_nums:
                    selected.add(fixture_test_nums[target_method])

            if any(
                mf.startswith(f"sqlglot/{d}")
                for d in ["dialects/", "parsers/", "generators/", "typing/"]
            ):
                has_dialect = True
            if mf.startswith("sqlglot/optimizer/"):
                has_optimizer = True

            if mf in FILE_TO_TESTS:
                selected.update(FILE_TO_TESTS[mf])
            else:
                for mapped_file, t_nums in FILE_TO_TESTS.items():
                    if mapped_file.endswith(mf) or mf.endswith(mapped_file):
                        selected.update(t_nums)

        if has_dialect:
            selected.update(core_dialect_nums)
        if has_optimizer:
            selected.update(optimizer_fast_nums)

        # Only run heavy 21.4s tpcds executor test if executor engine, serde, or eliminate_joins touched
        needs_heavy_exec = any(
            x in patch_text
            for x in [
                "sqlglot/executor/runner.py",
                "sqlglot/executor/python.py",
                "serde.py",
                "eliminate_joins.py",
            ]
        )
        if needs_heavy_exec and tpcds_exec_num:
            selected.add(tpcds_exec_num)
        elif tpcds_exec_num and tpcds_exec_num in selected:
            selected.remove(tpcds_exec_num)

        # Exclude slow memory test (6.7s) unless generator.py core touched
        if (
            mem_test_num
            and "sqlglot/generator.py" not in patch_text
            and mem_test_num in selected
        ):
            selected.remove(mem_test_num)

        # Exclude slow max_nodes recursion test (28.8s) unless parser.py core touched
        if (
            max_nodes_num
            and "sqlglot/parser.py" not in patch_text
            and max_nodes_num in selected
        ):
            selected.remove(max_nodes_num)

        workers = pack_tests(sorted(selected), tests_by_num)
        print(json.dumps({"commit": cid, "workers": workers}))

    print(
        f"Completed test selection for {len(list(commits_dir.glob('*.patch')))} commits",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
