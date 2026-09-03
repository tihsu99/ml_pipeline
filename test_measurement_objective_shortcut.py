import unittest

import yaml

from scripts.train_neutrino_backend import (
    MEASUREMENT_OVERLAY,
    MEASUREMENT_SDM_OVERLAY,
    default_overlay_for_backend,
)


class MeasurementObjectiveShortcutTests(unittest.TestCase):
    def test_default_remains_cdiag(self):
        self.assertEqual(
            default_overlay_for_backend("evenet-align"),
            MEASUREMENT_OVERLAY,
        )

    def test_explicit_sdm_mode_selects_valid_sdm_overlay(self):
        self.assertEqual(
            default_overlay_for_backend("evenet-align", "sdm_frobenius"),
            MEASUREMENT_SDM_OVERLAY,
        )
        strategy = yaml.safe_load(MEASUREMENT_SDM_OVERLAY.read_text())[
            "options"
        ]["Training"]["strategy"]
        self.assertEqual(strategy["objective"]["mode"], "sdm_frobenius")
        self.assertEqual(strategy["selection_metric"], "val/J_sdm")


if __name__ == "__main__":
    unittest.main()
