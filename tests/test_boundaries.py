"""Enforces the privacy boundary in code: coordinator modules may not touch simulator ground truth.

This passes trivially until src/privatefair/coordinator/ exists, then guards it forever.
"""

import ast
from pathlib import Path

COORDINATOR_DIR = Path(__file__).resolve().parents[1] / "src" / "privatefair" / "coordinator"
FORBIDDEN_NAMES = {"TrueBins", "sim_only"}
FORBIDDEN_MODULES = ("privatefair.sim", "privatefair.data")


def _python_files():
    return sorted(COORDINATOR_DIR.rglob("*.py")) if COORDINATOR_DIR.exists() else []


def test_coordinator_never_sees_ground_truth():
    violations = []
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(FORBIDDEN_MODULES):
                    violations.append(f"{path.name}: imports {node.module}")
                violations += [f"{path.name}: imports {a.name}" for a in node.names if a.name in FORBIDDEN_NAMES]
            elif isinstance(node, ast.Import):
                violations += [
                    f"{path.name}: imports {a.name}" for a in node.names if a.name.startswith(FORBIDDEN_MODULES)
                ]
            elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                violations.append(f"{path.name}:{node.lineno} uses {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
                violations.append(f"{path.name}:{node.lineno} uses .{node.attr}")
    assert not violations, "Privacy boundary violated:\n" + "\n".join(violations)
