"""Hard rules 1 and 2 (PLAN: repository layout and module contracts).

1. Nothing under qatester/agents/ may import ground truth.
2. Nothing under qatester/oracles/ may import vizdoom.

Checked from day one so the rule exists before the code it guards.
"""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "qatester"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _violations(subpkg: str, forbidden: tuple[str, ...]) -> list[str]:
    found = []
    for path in sorted((PKG / subpkg).rglob("*.py")):
        for mod in _imported_modules(path):
            if any(mod == f or mod.startswith(f + ".") for f in forbidden):
                found.append(f"{path.relative_to(PKG.parent)} imports {mod}")
    return found


def test_agents_do_not_import_groundtruth() -> None:
    assert _violations("agents", ("qatester.mapgen.groundtruth", "qatester.groundtruth")) == []


def test_oracles_do_not_import_vizdoom() -> None:
    assert _violations("oracles", ("vizdoom",)) == []
