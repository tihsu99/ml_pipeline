from __future__ import annotations

import unittest

from evaluation_config import post_calibration_enabled, qi_region_to_signals


class PostCalibrationConfigTest(unittest.TestCase):
    def test_defaults_to_enabled(self) -> None:
        self.assertTrue(post_calibration_enabled({}))

    def test_explicit_false_disables_calibration(self) -> None:
        self.assertFalse(
            post_calibration_enabled({"EveNetEvaluation": {"post_calibration": False}})
        )

    def test_rejects_non_boolean_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be true or false"):
            post_calibration_enabled({"EveNetEvaluation": {"post_calibration": "false"}})

    def test_qi_region_mapping_preserves_combined_channels(self) -> None:
        mapping = qi_region_to_signals(
            {
                "QIAnalysis": {
                    "region_to_signals": {
                        "Ztautau_pie": ["Ztautau_pie", "Ztautau_epi"],
                    }
                }
            }
        )
        self.assertEqual(mapping["Ztautau_pie"], ["Ztautau_pie", "Ztautau_epi"])

    def test_qi_region_mapping_rejects_empty_signals(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty signal list"):
            qi_region_to_signals(
                {"QIAnalysis": {"region_to_signals": {"Ztautau_pie": []}}}
            )

    def test_qi_region_mapping_defaults_to_one_signal_per_region(self) -> None:
        self.assertEqual(
            qi_region_to_signals({}, ["Ztautau_pie", "Ztautau_epi"]),
            {
                "Ztautau_pie": ["Ztautau_pie"],
                "Ztautau_epi": ["Ztautau_epi"],
            },
        )


if __name__ == "__main__":
    unittest.main()
