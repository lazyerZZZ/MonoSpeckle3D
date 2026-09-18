import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pseudo_overlap.data import discover_raw_triplets, group_split, tile_triplets


class DataPipelineTest(unittest.TestCase):
    def test_group_split_has_no_overlap(self):
        split = group_split((str(index) for index in range(10)), seed=7)
        groups = [set(values) for values in split.values()]
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])
        self.assertEqual(set.union(*groups), {str(index) for index in range(10)})

    def test_triplets_are_tiled_in_alignment(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            output = Path(temp) / "tiles"
            source.mkdir()
            for index, value in enumerate((20, 80, 140)):
                Image.new("L", (512, 256), value).save(source / f"{index}.bmp")
            triplets = discover_raw_triplets(source)
            self.assertEqual(tile_triplets(triplets, output), 2)
            self.assertEqual(len(list(output.glob("*.png"))), 6)
            with Image.open(output / "1_2_clear.png") as image:
                self.assertEqual(image.getpixel((0, 0)), 80)


if __name__ == "__main__":
    unittest.main()
