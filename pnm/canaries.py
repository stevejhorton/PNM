from __future__ import annotations
import torch, json, pathlib
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class CanarySuite:
    inputs: torch.Tensor
    expected: torch.Tensor
    atol: float = 1e-3

    @staticmethod
    def make_linear_suite(n: int = 16, low: float = -1.0, high: float = 1.0):
        xs = torch.linspace(low, high, n).view(-1, 1)
        # For ground-truth y = 2x + 1 (same as training target)
        ys = 2 * xs + 1
        return CanarySuite(inputs=xs, expected=ys, atol=5e-2)

    def save(self, path: str):
        p = pathlib.Path(path)
        p.write_text(json.dumps({
            "inputs": self.inputs.tolist(),
            "expected": self.expected.tolist(),
            "atol": self.atol
        }, indent=2))

    @staticmethod
    def load(path: str) -> "CanarySuite":
        d = json.loads(pathlib.Path(path).read_text())
        return CanarySuite(inputs=torch.tensor(d["inputs"], dtype=torch.float32),
                           expected=torch.tensor(d["expected"], dtype=torch.float32),
                           atol=float(d.get("atol", 1e-3)))

    def evaluate(self, model) -> Tuple[bool, float]:
        with torch.no_grad():
            y = model(self.inputs)
            err = torch.max(torch.abs(y - self.expected)).item()
            return (err <= self.atol), err
