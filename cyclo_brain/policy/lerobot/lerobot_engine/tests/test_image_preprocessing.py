#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "image_preprocessing.py"
spec = importlib.util.spec_from_file_location("image_preprocessing", MODULE_PATH)
image_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(image_preprocessing)


class ImagePreprocessingTest(unittest.TestCase):
    def test_rotates_wrist_image_from_640x480_to_480x640(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        rotated = image_preprocessing.apply_rotation(image, 270)

        self.assertEqual(rotated.shape, (640, 480, 3))

    def test_prepare_policy_image_does_not_resize(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        prepared = image_preprocessing.prepare_policy_image(image)

        self.assertEqual(prepared.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()
