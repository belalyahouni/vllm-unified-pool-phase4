import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize_trace.py"


def load_module():
    spec = importlib.util.spec_from_file_location("summarize_trace", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRACE = """\
UNIFIED CACHE L0 step=100 F=96 expert_sb 40/67 ours (expert-ours-sb=40, expert-other-sb=2, prefix-pages=1200, alloc-kv-pages=1, pinned-sb=0, hits=80, misses=20, ever-activated=64)
UNIFIED CACHE L15 step=115 F=96 expert_sb 8/67 ours (expert-ours-sb=8, expert-other-sb=34, prefix-pages=1200, alloc-kv-pages=1, pinned-sb=0, hits=95, misses=5, ever-activated=8)
UNIFIED DECISION side=kv-alloc step=116 layer=all expert-score=70.000 kv-score=80.000 chosen=expert
UNIFIED DECISION side=expert-miss step=117 layer=0 expert-score=72.000 kv-score=81.000 chosen=expert
UNIFIED DECISION side=expert-miss step=118 layer=15 expert-score=100.000 kv-score=82.000 chosen=kv
UNIFIED EVICT sb=4 L0 kind=expert E7 cause=kv-alloc-evict-expert tier=kv-broadcast score=70.000 step=116
"""


class SummaryTests(unittest.TestCase):
    def test_reports_layer_outcomes_and_decisions(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.log"
            path.write_text(TRACE)
            report = module.summarize(path)

        self.assertIn("## Per-layer outcomes", report)
        self.assertIn("| L0 | 40 | 64 | 80 | 20 | 80.0% | 1 |", report)
        self.assertIn("| L15 | 8 | 8 | 95 | 5 | 95.0% | 0 |", report)
        self.assertIn("## Mixed-LRU decisions", report)
        self.assertIn("| `expert-miss` | `expert` | 1 |", report)
        self.assertIn("| `expert-miss` | `kv` | 1 |", report)
        self.assertIn("| `kv-alloc` | `expert` | 1 |", report)
        self.assertIn("| L0 | 1 | 1 | 0 |", report)
        self.assertIn("| L15 | 1 | 0 | 1 |", report)


if __name__ == "__main__":
    unittest.main()
