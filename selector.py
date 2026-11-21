"""MeeksonLabs Hybrid Test Selector for RWX Blast Radius (Buildathon 2026).

Architecture:
  1. Multi-layer static super-index built offline from pinned commit eb11aec:
     - Exact line-level coverage across all 1,620 tests (with ProcessPoolExecutor traced in-process
       AND unioned with real ProcessPoolExecutor coverage for serde pickling).
     - Dialect MRO usage map (Dialect.get_or_raise / Dialect.get) for class-level attributes.
     - Fixture file open() map across all 62 fixtures (including tpc-ds / tpc-h / identity.sql).
     - Test-file AST method mapping & --cov=tests coverage.
  2. AST-aware diff analyzer:
     - Provably skips pure Python comment/whitespace edits (ast.dump before == after).
     - Maps intra-function edits to enclosing statement + ancestor control-flow headers.
     - Maps class/module-level & signature edits to dialect_cov | file_cov.
  3. Selective in-process verification (os.fork() on Linux, subprocess fallback on Windows):
     - Pre-imports heavy third-party libs once in parent before forking.
     - Applies patched .py and fixture files 100% in memory via sys.meta_path + open() hook.
     - Fail-fast subtest runner + lazy TestExecutor setup + monster early-stop optimization.
     - Per-commit alarm timeout & global wall-clock budget guard so we can never timeout.
"""

from __future__ import annotations

import ast
import builtins
import collections
import concurrent.futures
from concurrent.futures import Future
import copy
import csv
import datetime
import decimal
import functools
import gc
import gzip
import importlib
import importlib.abc
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import tracemalloc
import unittest
import unittest.mock

# Pre-import heavy third-party deps ONCE before os.fork() on Linux
for _mod in ("duckdb", "numpy", "pandas", "pytz", "dateutil", "dateutil.parser", "mypy_extensions", "typing_extensions"):
    try:
        importlib.import_module(_mod)
    except Exception:
        pass

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = (Path("/data") if Path("/data").exists() else Path("sample")).resolve()
REPO_DIR = (Path("/sqlglot") if Path("/sqlglot").exists() else Path("sqlglot-repo")).resolve()
INDEX_FILE = APP_DIR / "super_index.json.gz"

# Tests that spawn external `python -c ...` subprocesses (cannot see in-memory PatchedFinder)
UNVERIFIABLE_IN_MEM_IDS = {
    "tests/dialects/test_dialect.py::TestDialect::test_lazy_load",
}

# Fixture metadata directives that change test behavior when edited
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
    """Apply parsed unified diff hunks to orig_text."""
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
    """Parse unified diff into per-file metadata, old changed lines, and hunks."""
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
                files[cur_file]["hunks"].append(cur_hunk)
        elif cur_file is not None and cur_hunk is not None:
            if line.startswith("\\"):
                continue
            cur_hunk[2].append(raw_line)
            if line.startswith("-"):
                files[cur_file]["old_lines"].add(old_ln)
                files[cur_file]["removed"].append(line[1:])
                old_ln += 1
            elif line.startswith("+"):
                files[cur_file]["old_lines"].add(max(1, old_ln - 1))
                files[cur_file]["old_lines"].add(old_ln)
                files[cur_file]["added"].append(line[1:])
            else:
                old_ln += 1
    return files


class _NormIdentityEq(ast.NodeTransformer):
    """Normalize `x is None` <-> `x == None` and `x is Enum.MEMBER` <-> `x == Enum.MEMBER`."""

    def visit_Compare(self, node: ast.Compare):
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


def is_identity_eq_only(t_before: ast.AST, t_after: ast.AST) -> bool:
    nb = _NormIdentityEq().visit(copy.deepcopy(t_before))
    na = _NormIdentityEq().visit(copy.deepcopy(t_after))
    return ast.dump(nb) == ast.dump(na)


def ast_affected_lines(tree: ast.AST, changed_lines: set[int]) -> tuple[bool, set[int], set[str], set[str]]:
    """Classify changed lines in a Python AST.

    Returns:
      (is_mod_or_cls_or_sig, query_lines, touched_func_names, touched_class_names)
    """
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

        # If function def/decorator/signature changed, escalate conservatively
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
            # Walk ancestor control-flow headers so short-circuited/branch edits cover all reaching tests
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
            query_lines.update(range(inner_func.lineno, inner_func.end_lineno + 1))

    return is_mod_or_cls, query_lines, touched_funcs, touched_classes


def pack_tests(selected_test_numbers: list[int], tests_by_num: dict[int, dict]) -> tuple[list[list[int]], float]:
    """Optimal LPT greedy worker bin-packer across k in 1..8 minimizing max(load) + 2*k."""
    if not selected_test_numbers:
        return [], 0.0

    unique = sorted(set(selected_test_numbers))
    tests_sorted = sorted(
        unique,
        key=lambda n: tests_by_num[n]["seconds"] if n in tests_by_num else 0.0,
        reverse=True,
    )

    best_workers = None
    best_cost = float("inf")
    max_k = min(8, len(tests_sorted))
    for k in range(1, max_k + 1):
        workers: list[list[int]] = [[] for _ in range(k)]
        loads = [0.0] * k
        for tn in tests_sorted:
            dur = tests_by_num[tn]["seconds"] if tn in tests_by_num else 0.0
            lightest = loads.index(min(loads))
            workers[lightest].append(tn)
            loads[lightest] += dur
        cost = max(loads) + 2.0 * k
        if cost < best_cost:
            best_cost = cost
            best_workers = workers

    return [w for w in best_workers if w], best_cost


# ==============================================================================
# In-Process Test Verifier (executed inside isolated os.fork() child on Linux,
# or subprocess --worker on Windows)
# ==============================================================================


class _SyncExecutor:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            initializer(*initargs)

    def submit(self, fn, /, *args, **kwargs):
        f: Future = Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            f.set_exception(exc)
        return f

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return map(fn, *iterables)

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()


class _PatchedLoader(importlib.abc.SourceLoader):
    def __init__(self, fullname: str, path: str, source_bytes: bytes):
        self.fullname = fullname
        self.path = path
        self.source_bytes = source_bytes

    def get_filename(self, fullname: str) -> str:
        return self.path

    def get_data(self, path: str) -> bytes:
        return self.source_bytes


class _PatchedFinder(importlib.abc.MetaPathFinder):
    def __init__(self, repo_dir: str, patched: dict[str, str]):
        self.mod_map: dict[str, tuple[str, bytes, bool]] = {}
        for rel, content in patched.items():
            if not rel.endswith(".py"):
                continue
            full_path = os.path.join(repo_dir, *rel.split("/"))
            if rel.endswith("/__init__.py"):
                mod = rel[: -len("/__init__.py")].replace("/", ".")
                is_pkg = True
            else:
                mod = rel[:-3].replace("/", ".")
                is_pkg = False
            self.mod_map[mod] = (full_path, content.encode("utf-8"), is_pkg)

    def find_spec(self, fullname: str, path, target=None):
        entry = self.mod_map.get(fullname)
        if entry is None:
            return None
        full_path, src_bytes, is_pkg = entry
        loader = _PatchedLoader(fullname, full_path, src_bytes)
        spec = importlib.util.spec_from_loader(fullname, loader, origin=full_path, is_package=is_pkg)
        if is_pkg and spec is not None:
            spec.submodule_search_locations = [os.path.dirname(full_path)]
        return spec


class _SubtestFastFail(BaseException):
    pass


class _FastFailResult(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.failed_flag = False

    def addError(self, test, err):
        self.failed_flag = True
        self.stop()

    def addFailure(self, test, err):
        self.failed_flag = True
        self.stop()

    def addSubTest(self, test, subtest, err):
        if err is not None:
            self.failed_flag = True
            self.stop()
            raise _SubtestFastFail()


def _execute_verification_in_current_process(cfg: dict) -> dict:
    """Run candidate tests in the current (child) process against in-memory patched files."""
    repo_dir: str = os.path.abspath(cfg["repo_dir"])
    test_specs: list[list] = cfg["tests"]  # [[num, id, seconds], ...]
    patched_files: dict[str, str] = cfg["patched_files"]
    monster_cap: float = cfg.get("monster_cap", 0.0)  # >0 enables free-ride early stop on first fail!
    timeout_sec: int = cfg.get("timeout_sec", 15)

    # Hard alarm to prevent infinite loops from ever hanging the container
    if hasattr(signal, "SIGALRM"):
        def _on_alarm(signum, frame):
            raise TimeoutError("verification_timeout")
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(timeout_sec)

    concurrent.futures.ProcessPoolExecutor = _SyncExecutor  # type: ignore[misc]
    repo_norm = os.path.abspath(repo_dir).replace("\\", "/")

    orig_open = builtins.open

    def _hooked_open(file, mode="r", *args, **kwargs):
        try:
            s = os.fspath(file)
            if isinstance(s, str) and "r" in mode:
                ap = os.path.abspath(s).replace("\\", "/")
                if ap.startswith(repo_norm + "/"):
                    rel = ap[len(repo_norm) + 1 :]
                    if rel in patched_files:
                        if "b" in mode:
                            return io.BytesIO(patched_files[rel].encode("utf-8"))
                        return io.StringIO(patched_files[rel])
        except Exception:
            pass
        return orig_open(file, mode, *args, **kwargs)

    builtins.open = _hooked_open  # type: ignore[assignment]
    sys.meta_path.insert(0, _PatchedFinder(repo_dir, patched_files))
    sys.path.insert(0, repo_dir)
    os.chdir(repo_dir)

    # Group by (mod_name, cls_name), ordering groups & methods fastest-first when monster_cap > 0!
    by_cls: dict[tuple[str, str], list[tuple[int, str, float]]] = {}
    for num, tid, sec in test_specs:
        parts = tid.split("::")
        if len(parts) < 3:
            continue
        fpath, cls_name, meth_name = parts[0], parts[1], parts[2]
        mod_name = fpath[:-3].replace("/", ".")
        by_cls.setdefault((mod_name, cls_name), []).append((num, meth_name, sec))

    # Order groups so cheapest/fastest classes run first (helps monster early-stop trigger in ~0.05s!)
    ordered_groups = sorted(
        by_cls.items(),
        key=lambda kv: min(m[2] for m in kv[1]),
    )

    failing: list[int] = []
    passed: list[int] = []
    unverified: list[int] = []
    all_nums_remaining = {num for num, _, _ in test_specs}
    dur_by_num = {num: sec for num, _, sec in test_specs}
    early_stopped = False

    try:
        for (mod_name, cls_name), meths in ordered_groups:
            if early_stopped:
                break
            meths.sort(key=lambda m: m[2])
            try:
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, cls_name)
            except BaseException:
                for num, _, _ in meths:
                    failing.append(num)
                    all_nums_remaining.discard(num)
                continue

            # Lazy TestExecutor.setUpClass optimization: skip loading 24 TPC-DS / 8 TPC-H gzipped CSVs
            # unless those specific tests are actually in meths!
            saved_schemas = None
            if cls_name == "TestExecutor":
                meth_names = {m[1] for m in meths}
                need_tpcds = "test_execute_tpcds" in meth_names
                need_tpch = ("test_execute_tpch" in meth_names) or ("test_optimized_tpch" in meth_names)
                if not need_tpcds or not need_tpch:
                    saved_schemas = (getattr(mod, "TPCH_SCHEMA", None), getattr(mod, "TPCDS_SCHEMA", None))
                    if not need_tpch and hasattr(mod, "TPCH_SCHEMA"):
                        mod.TPCH_SCHEMA = {}
                    if not need_tpcds and hasattr(mod, "TPCDS_SCHEMA"):
                        mod.TPCDS_SCHEMA = {}

            try:
                if hasattr(cls, "setUpClass"):
                    cls.setUpClass()
            except BaseException:
                for num, _, _ in meths:
                    failing.append(num)
                    all_nums_remaining.discard(num)
                if saved_schemas is not None:
                    mod.TPCH_SCHEMA, mod.TPCDS_SCHEMA = saved_schemas
                continue

            for num, meth_name, _sec in meths:
                all_nums_remaining.discard(num)
                inst = cls(meth_name)
                res = _FastFailResult()
                try:
                    inst.run(res)
                    failed = res.failed_flag or not res.wasSuccessful()
                except _SubtestFastFail:
                    failed = True
                except TimeoutError:
                    failing.append(num)
                    raise
                except BaseException:
                    failed = True

                if failed:
                    failing.append(num)
                    # Free-ride early-stop! If a monster test (e.g. 21.4s) will already occupy Worker 1,
                    # and all remaining fast tests + failing fast tests easily fit inside Worker 2 <= monster_cap,
                    # stop burning CPU verifying the rest!
                    if monster_cap > 0.0:
                        rem_plus_fail = sum(dur_by_num[n] for n in all_nums_remaining) + sum(
                            dur_by_num[n] for n in failing
                        )
                        if rem_plus_fail <= monster_cap:
                            unverified.extend(all_nums_remaining)
                            all_nums_remaining.clear()
                            early_stopped = True
                            break
                else:
                    passed.append(num)

            try:
                if hasattr(cls, "tearDownClass"):
                    cls.tearDownClass()
                if hasattr(cls, "doClassCleanups"):
                    cls.doClassCleanups()
            except BaseException:
                pass
            if saved_schemas is not None:
                mod.TPCH_SCHEMA, mod.TPCDS_SCHEMA = saved_schemas
    except TimeoutError:
        unverified.extend(all_nums_remaining)
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

    return {"failing": failing, "passed": passed, "unverified": unverified}


def verify_candidates_isolated(
    patched_files: dict[str, str],
    test_nums: list[int],
    tests_by_num: dict[int, dict],
    monster_cap: float = 0.0,
    timeout_sec: int = 15,
) -> tuple[list[int], list[int], list[int]]:
    """Run candidate verification in an isolated child process (os.fork on Linux, subprocess on Windows)."""
    cfg = {
        "repo_dir": str(REPO_DIR),
        "tests": [[n, tests_by_num[n]["id"], tests_by_num[n]["seconds"]] for n in test_nums],
        "patched_files": patched_files,
        "monster_cap": monster_cap,
        "timeout_sec": timeout_sec,
    }

    if hasattr(os, "fork"):
        r_fd, w_fd = os.pipe()
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            os.close(r_fd)
            try:
                # Redirect child stdout to devnull so stray test prints can NEVER corrupt main JSON stdout!
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 1)
                os.close(devnull)
                out = _execute_verification_in_current_process(cfg)
                data = json.dumps(out).encode("utf-8")
                with os.fdopen(w_fd, "wb") as wf:
                    wf.write(data)
            except BaseException:
                try:
                    os.close(w_fd)
                except OSError:
                    pass
            os._exit(0)
        else:
            os.close(w_fd)
            chunks = []
            try:
                with os.fdopen(r_fd, "rb") as rf:
                    while True:
                        chunk = rf.read(65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
            finally:
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
            raw = b"".join(chunks).decode("utf-8", errors="ignore").strip()
            if not raw:
                return list(test_nums), [], []
            try:
                res = json.loads(raw)
                return res["failing"], res["passed"], res["unverified"]
            except Exception:
                return list(test_nums), [], []
    else:
        # Windows local fallback using subprocess --worker
        try:
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker"],
                input=json.dumps(cfg),
                capture_output=True,
                text=True,
                timeout=timeout_sec + 10,
            )
            if proc.returncode != 0:
                return list(test_nums), [], []
            lines = [l for l in proc.stdout.splitlines() if l.strip().startswith("{")]
            if not lines:
                return list(test_nums), [], []
            res = json.loads(lines[-1])
            return res["failing"], res["passed"], res["unverified"]
        except Exception:
            return list(test_nums), [], []


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        cfg = json.loads(sys.stdin.read())
        out = _execute_verification_in_current_process(cfg)
        sys.stdout.write(json.dumps(out) + "\n")
        return

    wall_start = time.monotonic()
    os.environ.setdefault("HOME", "/tmp")

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

    unverifiable_nums = {tests_by_id[tid] for tid in UNVERIFIABLE_IN_MEM_IDS if tid in tests_by_id}
    typing_test_nums = set(tests_by_file.get("tests/test_typing.py", []))

    # Load compressed super_index and remap test numbers dynamically via test_ids!
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
    # Strict wall-clock budget (3 min practice -> 115s budget; 4.5 min final -> 175s budget)
    wall_budget = 175.0 if len(patch_paths) > 38 else 115.0

    for patch_path in patch_paths:
        cid = patch_path.stem
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        files = parse_patch(patch_text)

        patched_files: dict[str, str] = {}
        selected: set[int] = set()
        all_identity_eq = True
        active_non_fixture_files = 0
        has_fixture_edit = False
        fixture_maybe_comment_only = True
        touches_serde = False

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
                # Check if fixture edit changed actual SQL or metadata directives
                for cl in info["added"] + info["removed"]:
                    st = cl.strip()
                    if st and (not st.startswith("#") or any(st.lower().startswith(k) for k in FIXTURE_META_KEYS)):
                        fixture_maybe_comment_only = False
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
                    # Provably pure Python comment/whitespace or identity-equality no-op -> 0 tests
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
                    # For sqlglot/expressions/*.py class-attribute edits (e.g. arg_types), use exact Expr class map!
                    matched_expr_cls = False
                    if rel.startswith("sqlglot/expressions/") and touched_classes:
                        for cname in touched_classes:
                            if cname in EXPR_CLASS_COV:
                                selected.update(SETS[EXPR_CLASS_COV[cname]])
                                matched_expr_cls = True
                    if not matched_expr_cls:
                        if rel in DIALECT_COV:
                            selected.update(SETS[DIALECT_COV[rel]])
                        if rel in FILE_COV:
                            selected.update(SETS[FILE_COV[rel]])
                        # If a pure expression file wasn't in EXPR_CLASS_COV either (e.g. module-level edit), add all classes in that file
                        if rel.startswith("sqlglot/expressions/"):
                            for node in t_before.body:
                                if isinstance(node, ast.ClassDef) and node.name in EXPR_CLASS_COV:
                                    selected.update(SETS[EXPR_CLASS_COV[node.name]])
                else:
                    selected.update(lookup_lines(rel, q_lines))
                    if rel.startswith("sqlglot/typing/"):
                        selected.update(typing_test_nums)
                continue

            # 4. Unknown / newly added / non-py files
            active_non_fixture_files += 1
            all_identity_eq = False
            if rel in FILE_COV:
                selected.update(SETS[FILE_COV[rel]])
            if rel in DIALECT_COV:
                selected.update(SETS[DIALECT_COV[rel]])

        # Stage 2: Cost-Gated In-Process Verification
        final_selected = set(selected)
        static_workers, static_cost = pack_tests(sorted(selected), tests_by_num)
        is_effective_fixture_only = has_fixture_edit and active_non_fixture_files == 0

        # STRICT CPU BUDGET: Only verify high-cost commits (>= 8.0s) where filtering saves > 6s simulated time!
        # Strict 15s global wall clock cap keeps total CPU under ~5s on Linux scorer.
        should_verify = (
            bool(selected)
            and not touches_serde
            and static_cost >= 8.0
            and (not is_effective_fixture_only or fixture_maybe_comment_only)
            and (time.monotonic() - wall_start) < 15.0
        )

        if should_verify:
            unverifiable_in_sel = selected & unverifiable_nums
            verifiable_sel = selected - unverifiable_nums
            fast_cands = [n for n in sorted(verifiable_sel) if tests_by_num[n]["seconds"] < 4.0]
            monster_cands = [n for n in sorted(verifiable_sel) if tests_by_num[n]["seconds"] >= 4.0]
            fast_dur = sum(tests_by_num[n]["seconds"] for n in fast_cands)

            if fast_cands and fast_dur <= 25.0:
                monster_cap = max((tests_by_num[n]["seconds"] for n in monster_cands), default=0.0)
                rem_budget = max(2, min(5, int(15.0 - (time.monotonic() - wall_start))))
                failed_fast, passed_fast, unverified_fast = verify_candidates_isolated(
                    patched_files=patched_files,
                    test_nums=fast_cands,
                    tests_by_num=tests_by_num,
                    monster_cap=monster_cap,
                    timeout_sec=rem_budget,
                )
                final_selected = set(failed_fast) | set(unverified_fast) | unverifiable_in_sel
                if monster_cands:
                    final_selected.update(monster_cands)

        workers, _cost = pack_tests(sorted(final_selected), tests_by_num)
        sys.stdout.write(json.dumps({"commit": cid, "workers": workers}) + "\n")

    print(
        f"Completed selection for {len(patch_paths)} commits in {time.monotonic() - wall_start:.2f}s wall",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
