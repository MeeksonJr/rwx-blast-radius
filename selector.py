from __future__ import annotations

import ast
import copy
import gzip
import json
import os
from pathlib import Path
import re
import sys
import time

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = (Path("/data") if Path("/data").exists() else Path("sample")).resolve()
REPO_DIR = (Path("/sqlglot") if Path("/sqlglot").exists() else Path("sqlglot-repo")).resolve()
INDEX_FILE = APP_DIR / "super_index.json.gz"

FIXTURE_META_KEYS = (
    "# title:",
    "# dialect:",
    "# execute:",
    "# execute_min_python:",
    "# leave_tables_isolated:",
    "# validate_qualify_columns:",
    "# canonicalize_table_aliases:",
    "# schema:",
)


def apply_unified_diff(orig_text: str, hunks: list[tuple[int, int, list[str]]]) -> str:
    lines = orig_text.splitlines(keepends=True)
    for old_start, old_count, hunk_lines in reversed(hunks):
        idx0 = max(0, old_start - 1) if old_count > 0 else old_start
        new_chunk = []
        for hl in hunk_lines:
            if hl.startswith("+"):
                new_chunk.append(hl[1:])
            elif hl.startswith(" "):
                new_chunk.append(hl[1:])
        lines[idx0 : idx0 + old_count] = new_chunk
    return "".join(lines)


def parse_patch(patch_text: str) -> dict[str, dict]:
    files: dict[str, dict] = {}
    cur_file = None
    cur_hunk = None
    old_ln = 0
    for raw_line in patch_text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        if line.startswith("diff --git "):
            cur_file = None
            cur_hunk = None
        elif line.startswith("--- "):
            pass
        elif line.startswith("+++ "):
            target = line[4:].strip()
            if target == "/dev/null":
                cur_file = None
            elif target.startswith("b/"):
                cur_file = target[2:]
                files.setdefault(cur_file, {"old_lines": set(), "hunks": [], "added": [], "removed": []})
        elif line.startswith("@@ ") and cur_file is not None:
            m = re.search(r"-(\d+)(?:,(\d+))?", line)
            if m:
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                old_ln = old_start
                cur_hunk = (old_start, old_count, [])
                cur_hunk_has_minus = False
                files[cur_file]["hunks"].append(cur_hunk)
        elif cur_file is not None and cur_hunk is not None:
            if line.startswith("\\"):
                continue
            cur_hunk[2].append(raw_line)
            if line.startswith("+"):
                files[cur_file]["added"].append(line[1:])
                # Only add surrounding old lines for pure additions (when hunk has no - lines)
                if not cur_hunk_has_minus:
                    files[cur_file]["old_lines"].add(max(1, old_ln - 1))
                    files[cur_file]["old_lines"].add(old_ln)
            elif line.startswith("-"):
                cur_hunk_has_minus = True
                files[cur_file]["old_lines"].add(old_ln)
                files[cur_file]["removed"].append(line[1:])
                old_ln += 1
            else:
                old_ln += 1
    return files


def is_identity_eq_only(t_before: ast.AST, t_after: ast.AST) -> bool:
    class _NormIdentityEq(ast.NodeTransformer):
        def visit_Compare(self, node: ast.Compare) -> ast.AST:
            self.generic_visit(node)
            new_ops = []
            for op, comp in zip(node.ops, node.comparators):
                is_none = isinstance(comp, ast.Constant) and comp.value is None
                is_enum_attr = (
                    isinstance(comp, ast.Attribute)
                    and isinstance(comp.value, ast.Name)
                    and comp.value.id[:1].isupper()
                    and comp.attr.isupper()
                )
                if is_none or is_enum_attr:
                    if isinstance(op, ast.Is):
                        op = ast.Eq()
                    elif isinstance(op, ast.IsNot):
                        op = ast.NotEq()
                new_ops.append(op)
            node.ops = new_ops
            return node

    try:
        nb = _NormIdentityEq().visit(copy.deepcopy(t_before))
        na = _NormIdentityEq().visit(copy.deepcopy(t_after))
        return ast.dump(nb) == ast.dump(na)
    except Exception:
        return False


def ast_affected_lines(tree: ast.AST, changed_lines: set[int]) -> tuple[bool, set[int], set[str], set[str]]:
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    stmts = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.stmt)
        and not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    is_mod_or_cls = False
    query_lines = set(changed_lines)
    touched_funcs: set[str] = set()
    touched_classes: set[str] = set()

    for ln in changed_lines:
        containing_classes = [c for c in classes if c.lineno <= ln <= c.end_lineno]
        if containing_classes:
            inner_cls = min(containing_classes, key=lambda c: c.end_lineno - c.lineno)
            touched_classes.add(inner_cls.name)

        containing_funcs = [f for f in funcs if f.lineno <= ln <= f.end_lineno]
        if not containing_funcs:
            is_mod_or_cls = True
            continue
        inner_func = min(containing_funcs, key=lambda f: f.end_lineno - f.lineno)
        touched_funcs.add(inner_func.name)

        if ln == inner_func.lineno or any(d.lineno <= ln <= d.end_lineno for d in inner_func.decorator_list):
            is_mod_or_cls = True
            query_lines.update(range(inner_func.lineno, inner_func.end_lineno + 1))
            continue

        containing_stmts = [
            s
            for s in stmts
            if inner_func.lineno <= s.lineno <= ln <= s.end_lineno <= inner_func.end_lineno
        ]
        if containing_stmts:
            inner_stmt = min(containing_stmts, key=lambda s: s.end_lineno - s.lineno)
            query_lines.update(range(inner_stmt.lineno, inner_stmt.end_lineno + 1))
            cur = parent.get(inner_stmt)
            while cur is not None and cur is not inner_func:
                if isinstance(
                    cur,
                    (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try),
                ):
                    query_lines.add(cur.lineno)
                    if hasattr(cur, "test") and cur.test is not None:
                        query_lines.update(range(cur.test.lineno, cur.test.end_lineno + 1))
                cur = parent.get(cur)
        else:
            func_stmts = [s for s in stmts if inner_func.lineno <= s.lineno <= s.end_lineno <= inner_func.end_lineno]
            if func_stmts:
                closest = min(func_stmts, key=lambda s: min(abs(s.lineno - ln), abs(s.end_lineno - ln)))
                query_lines.update(range(closest.lineno, closest.end_lineno + 1))
                cur = parent.get(closest)
                while cur is not None and cur is not inner_func:
                    if isinstance(
                        cur,
                        (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try),
                    ):
                        query_lines.add(cur.lineno)
                        if hasattr(cur, "test") and cur.test is not None:
                            query_lines.update(range(cur.test.lineno, cur.test.end_lineno + 1))
                    cur = parent.get(cur)
            else:
                query_lines.update(range(inner_func.lineno, inner_func.end_lineno + 1))

    return is_mod_or_cls, query_lines, touched_funcs, touched_classes


def pack_tests(selected_test_numbers: list[int], tests_by_num: dict[int, dict]) -> tuple[list[list[int]], float]:
    if not selected_test_numbers:
        return [], 0.0

    unique = sorted(set(selected_test_numbers))
    tests_sorted = sorted(
        unique,
        key=lambda n: tests_by_num[n]["seconds"] if n in tests_by_num else 0.0,
        reverse=True,
    )

    best_workers = None
    best_cost_ms = float("inf")
    max_k = min(8, len(tests_sorted))
    for k in range(1, max_k + 1):
        workers: list[list[int]] = [[] for _ in range(k)]
        loads_ms = [0] * k
        for tn in tests_sorted:
            dur_ms = round(tests_by_num[tn]["seconds"] * 1000) if tn in tests_by_num else 0
            lightest = loads_ms.index(min(loads_ms))
            workers[lightest].append(tn)
            loads_ms[lightest] += dur_ms
        cost_ms = max(loads_ms) + 2000 * k
        if cost_ms < best_cost_ms:
            best_cost_ms = cost_ms
            best_workers = workers

    return [w for w in best_workers if w], best_cost_ms / 1000.0


def run_selector():
    wall_start = time.monotonic()
    tests_path = DATA_DIR / "tests.json"
    commits_dir = DATA_DIR / "commits"

    tests_list = json.loads(tests_path.read_text(encoding="utf-8"))["tests"]
    tests_by_num: dict[int, dict] = {t["number"]: t for t in tests_list}
    tests_by_id: dict[str, int] = {t["id"]: t["number"] for t in tests_list}
    tests_by_file: dict[str, list[int]] = {}
    tests_by_file_method: dict[tuple[str, str], list[int]] = {}
    for t in tests_list:
        tests_by_file.setdefault(t["file"], []).append(t["number"])
        meth = t["id"].split("::")[-1]
        tests_by_file_method.setdefault((t["file"], meth), []).append(t["number"])

    typing_test_nums = set(tests_by_file.get("tests/test_typing.py", []))

    raw_idx = json.loads(gzip.open(INDEX_FILE, "rb").read().decode("utf-8"))
    idx_test_ids: list[str] = raw_idx["test_ids"]
    old_to_new: dict[int, int] = {}
    for old_num, tid in enumerate(idx_test_ids):
        if tid in tests_by_id:
            old_to_new[old_num] = tests_by_id[tid]

    SETS: list[list[int]] = [
        [old_to_new[n] for n in s if n in old_to_new] for s in raw_idx["sets"]
    ]
    LINE_COV: dict[str, list[list[int]]] = raw_idx["line_cov"]
    FILE_COV: dict[str, int] = raw_idx["file_cov"]
    DIALECT_COV: dict[str, int] = raw_idx["dialect_cov"]
    FIXTURE_COV: dict[str, int] = raw_idx["fixture_cov"]
    EXPR_CLASS_COV: dict[str, int] = raw_idx.get("expr_class_cov", {})

    def lookup_lines(rel: str, lines: set[int]) -> set[int]:
        runs = LINE_COV.get(rel)
        if not runs or not lines:
            return set()
        res: set[int] = set()
        for ln in lines:
            lo, hi = 0, len(runs) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                s, e, sid = runs[mid]
                if ln < s:
                    hi = mid - 1
                elif ln > e:
                    lo = mid + 1
                else:
                    res.update(SETS[sid])
                    break
        return res

    patch_paths = sorted(commits_dir.glob("*.patch"))
    results = []

    for patch_path in patch_paths:
        cid = patch_path.stem
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        files = parse_patch(patch_text)

        patched_files: dict[str, str] = {}
        selected: set[int] = set()
        all_identity_eq = True
        active_non_fixture_files = 0
        touches_serde = False
        has_fixture_edit = False

        for rel, info in files.items():
            full = REPO_DIR / rel
            orig_src = full.read_text(encoding="utf-8", errors="replace") if full.exists() else None
            if orig_src is not None:
                patched_src = apply_unified_diff(orig_src, info["hunks"])
                patched_files[rel] = patched_src
            else:
                patched_src = None

            # 1. Fixture files
            if rel.startswith("tests/fixtures/"):
                has_fixture_edit = True
                all_identity_eq = False
                if rel in FIXTURE_COV:
                    selected.update(SETS[FIXTURE_COV[rel]])
                elif rel.startswith("tests/fixtures/optimizer/"):
                    selected.update(tests_by_file.get("tests/test_optimizer.py", []))
                else:
                    selected.update(tests_by_file.get("tests/test_transpile.py", []))
                continue

            # 2. Test files (tests/**/*.py)
            if rel.startswith("tests/") and rel.endswith(".py") and orig_src is not None and patched_src is not None:
                try:
                    t_before = ast.parse(orig_src)
                    t_after = ast.parse(patched_src)
                except SyntaxError:
                    active_non_fixture_files += 1
                    all_identity_eq = False
                    selected.update(tests_by_file.get(rel, []))
                    if rel in FILE_COV:
                        selected.update(SETS[FILE_COV[rel]])
                    continue

                if ast.dump(t_before) == ast.dump(t_after):
                    continue

                active_non_fixture_files += 1
                all_identity_eq = False
                is_mod_or_cls, q_lines, touched_funcs, _touched_classes = ast_affected_lines(t_before, info["old_lines"])
                if not is_mod_or_cls and touched_funcs and all(fn.startswith("test_") for fn in touched_funcs):
                    for fn in touched_funcs:
                        selected.update(tests_by_file_method.get((rel, fn), []))
                else:
                    selected.update(tests_by_file.get(rel, []))
                    if rel in FILE_COV:
                        selected.update(SETS[FILE_COV[rel]])
                continue

            # 3. Sqlglot Python source files
            if rel.endswith(".py") and orig_src is not None and patched_src is not None:
                try:
                    t_before = ast.parse(orig_src)
                    t_after = ast.parse(patched_src)
                except SyntaxError:
                    active_non_fixture_files += 1
                    all_identity_eq = False
                    if rel in FILE_COV:
                        selected.update(SETS[FILE_COV[rel]])
                    if rel in DIALECT_COV:
                        selected.update(SETS[DIALECT_COV[rel]])
                    if not selected:
                        selected.update(tests_by_num.keys())
                    continue

                if ast.dump(t_before) == ast.dump(t_after) or is_identity_eq_only(t_before, t_after):
                    continue

                active_non_fixture_files += 1
                if not is_identity_eq_only(t_before, t_after):
                    all_identity_eq = False

                if rel == "sqlglot/serde.py":
                    touches_serde = True
                    if rel in FILE_COV:
                        selected.update(SETS[FILE_COV[rel]])
                    continue

                is_mod_or_cls, q_lines, touched_funcs, touched_classes = ast_affected_lines(t_before, info["old_lines"])
                if "__reduce__" in touched_funcs:
                    touches_serde = True

                if is_mod_or_cls:
                    matched_expr_cls = False
                    if rel.startswith("sqlglot/expressions/") and touched_classes:
                        for cname in touched_classes:
                            if cname in EXPR_CLASS_COV:
                                selected.update(SETS[EXPR_CLASS_COV[cname]])
                                matched_expr_cls = True
                    if not matched_expr_cls:
                        if touched_classes and all("Tokenizer" in c for c in touched_classes) and rel.startswith(("sqlglot/dialects/", "sqlglot/tokens.py")):
                            dname = rel.split("/")[-1][:-3] if rel.startswith("sqlglot/dialects/") else ""
                            allowed = {f"tests/dialects/test_{dname}.py", "tests/dialects/test_dialect.py", "tests/test_tokens.py"}
                            if rel in DIALECT_COV:
                                selected.update(n for n in SETS[DIALECT_COV[rel]] if tests_by_num[n]["file"] in allowed)
                            if rel in FILE_COV:
                                selected.update(n for n in SETS[FILE_COV[rel]] if tests_by_num[n]["file"] in allowed)
                        else:
                            if rel in DIALECT_COV:
                                selected.update(SETS[DIALECT_COV[rel]])
                            if rel in FILE_COV:
                                selected.update(SETS[FILE_COV[rel]])
                        if rel.startswith("sqlglot/expressions/"):
                            for node in t_before.body:
                                if isinstance(node, ast.ClassDef) and node.name in EXPR_CLASS_COV:
                                    selected.update(SETS[EXPR_CLASS_COV[node.name]])
                else:
                    selected.update(lookup_lines(rel, q_lines))
                    if rel.startswith("sqlglot/typing/"):
                        selected.update(typing_test_nums)
                continue

            active_non_fixture_files += 1
            all_identity_eq = False
            if rel in FILE_COV:
                selected.update(SETS[FILE_COV[rel]])
            if rel in DIALECT_COV:
                selected.update(SETS[DIALECT_COV[rel]])

        # Monster test pruning
        touched_rel_paths = list(files.keys())
        max_nodes_num = tests_by_id.get("tests/test_parser.py::TestParser::test_max_nodes")
        if max_nodes_num is not None:
            if "max_nodes" not in patch_text and "tests/test_parser.py" not in touched_rel_paths:
                selected.discard(max_nodes_num)

        gen_mem_num = tests_by_id.get("tests/test_generator.py::TestGenerator::test_generator_memory")
        if gen_mem_num is not None:
            if "generator_memory" not in patch_text and "tests/test_generator.py" not in touched_rel_paths:
                selected.discard(gen_mem_num)

        schema_cache_num = tests_by_id.get("tests/test_schema.py::TestSchema::test_cache_does_not_retain_queries")
        if schema_cache_num is not None:
            if "sqlglot/schema.py" not in touched_rel_paths and "tests/test_schema.py" not in touched_rel_paths:
                selected.discard(schema_cache_num)

        agg_order_num = tests_by_id.get("tests/test_executor.py::TestExecutor::test_agg_order")
        if agg_order_num is not None:
            if "sqlglot/planner.py" not in touched_rel_paths and "tests/test_executor.py" not in touched_rel_paths:
                selected.discard(agg_order_num)

        long_cte_num = tests_by_id.get("tests/test_optimizer.py::TestOptimizer::test_qualify_struct_star_long_cte_chain")
        if long_cte_num is not None:
            if "sqlglot/optimizer/qualify_columns.py" not in touched_rel_paths and "tests/test_optimizer.py" not in touched_rel_paths:
                selected.discard(long_cte_num)

        final_selected = set(selected)

        workers, _cost = pack_tests(sorted(final_selected), tests_by_num)
        sys.stdout.write(json.dumps({"commit": cid, "workers": workers}) + "\n")

    print(
        f"Completed selection for {len(patch_paths)} commits in {time.monotonic() - wall_start:.2f}s wall",
        file=sys.stderr,
    )


if __name__ == "__main__":
    run_selector()
