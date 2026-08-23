import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "recreate_reference_memory_figure.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "recreate_reference_memory_figure", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecreateReferenceFigureTests(unittest.TestCase):
    def test_writes_reference_sized_png(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.png"
            module.render(output)

            with Image.open(output) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (2786, 1488))

    def test_writes_seaborn_version_with_same_dimensions(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "seaborn.png"
            module.render_seaborn(output)

            with Image.open(output) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (2764, 1486))

    def test_writes_scienceplots_version(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scienceplots.png"
            module.render_scienceplots(output)

            with Image.open(output) as image:
                self.assertEqual(image.format, "PNG")


if __name__ == "__main__":
    unittest.main()
