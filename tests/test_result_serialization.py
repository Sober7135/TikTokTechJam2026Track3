import json
import math
import tempfile
import unittest
from pathlib import Path

from transformer_benchmark.runner import write_json_document


class ResultSerializationTests(unittest.TestCase):
    def test_non_finite_diagnostics_are_written_as_null(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json_document(
                path,
                {
                    "nan": math.nan,
                    "positive_infinity": math.inf,
                    "nested": [-math.inf, 1.0],
                },
            )

            result = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsNone(result["nan"])
            self.assertIsNone(result["positive_infinity"])
            self.assertEqual(result["nested"], [None, 1.0])


if __name__ == "__main__":
    unittest.main()
