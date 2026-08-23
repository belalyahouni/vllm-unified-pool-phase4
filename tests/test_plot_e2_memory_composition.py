import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "plot_e2_memory_composition.py"


def load_plot_module():
    spec = importlib.util.spec_from_file_location(
        "plot_e2_memory_composition", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_line(ours, other, prefix, allocated):
    return (
        "UNIFIED CACHE L0 step=1 F=96 expert_sb 4/67 ours "
        f"(expert-ours-sb={ours}, expert-other-sb={other}, "
        f"prefix-pages={prefix}, alloc-kv-pages={allocated}, "
        "pinned-sb=0, ever-activated=4)\n"
    )


GET = 'GET /metrics HTTP/1.1" 200 OK\n'
POST = 'POST /v1/completions HTTP/1.1" 200 OK\n'


class ParseTraceTests(unittest.TestCase):
    def setUp(self):
        self.module = load_plot_module()

    def write_trace(self):
        trace = "".join(
            [
                GET,
                cache_line(5, 1, 100, 5),
                POST,
                GET,
                GET,
                cache_line(4, 1, 110, 6),
                POST,
                GET,
                GET,
                cache_line(3, 1, 120, 8),
                POST,
                cache_line(4, 0, 120, 8),
                POST,
                GET,
            ]
        )
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "trace.log"
        path.write_text(trace)
        return directory, path

    def test_reconstructs_request_level_page_occupancy(self):
        directory, path = self.write_trace()
        self.addCleanup(directory.cleanup)

        data = self.module.parse_trace(path, "kv2exp")

        self.assertEqual(len(data.samples), 4)
        self.assertEqual(data.phase_request, 2)
        self.assertEqual(data.samples[-1].expert_pages, 384)
        self.assertEqual(data.samples[-1].kv_pages, 128)
        self.assertEqual(data.samples[-1].free_pages, 5920)

    def test_exp_to_kv_phase_follows_first_segment(self):
        directory, path = self.write_trace()
        self.addCleanup(directory.cleanup)

        data = self.module.parse_trace(path, "exp2kv")

        self.assertEqual(data.phase_request, 1)

    def test_rejects_occupancy_above_pool_capacity(self):
        trace = GET + cache_line(67, 0, 1, 0) + POST + GET
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "trace.log"
        path.write_text(trace)

        with self.assertRaisesRegex(ValueError, "pool capacity"):
            self.module.parse_trace(path, "kv2exp")


class RenderFigureTests(unittest.TestCase):
    def setUp(self):
        self.module = load_plot_module()

    def trace_data(self, phase_request):
        samples = [
            self.module.Sample(1, 4608, 1000, 824),
            self.module.Sample(2, 4000, 1800, 632),
            self.module.Sample(3, 5200, 900, 332),
        ]
        return self.module.TraceData(samples, phase_request)

    def test_writes_three_vector_pdf_variants(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        output_dir = Path(directory.name)

        outputs = self.module.render_variants(
            self.trace_data(2), self.trace_data(1), output_dir
        )

        self.assertEqual(len(outputs), 3)
        for output in outputs:
            self.assertTrue(output.exists())
            self.assertEqual(output.read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
