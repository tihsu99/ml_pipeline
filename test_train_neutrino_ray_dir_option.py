import unittest
from pathlib import Path

from scripts.train_neutrino_backend import command_for_backend


class TrainNeutrinoRayDirectoryTests(unittest.TestCase):
    def test_ray_directory_is_forwarded_with_backend_specific_spelling(self):
        runtime = Path("runtime.yaml")
        ray_dir = Path("results/ray")
        expected = {
            "pure-evenet": "--ray_dir",
            "dgpo-evenet": "--ray-dir",
            "evenet-align": "--ray_dir",
        }
        for backend, option in expected.items():
            with self.subTest(backend=backend):
                command = command_for_backend(backend, runtime, ray_dir=ray_dir)
                self.assertEqual(command[-2:], [option, str(ray_dir)])


if __name__ == "__main__":
    unittest.main()
