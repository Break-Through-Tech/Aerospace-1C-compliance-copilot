"""Ensure invalid citations, provenance, and labels fail validation."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('validator', ROOT / 'scripts/validate_benchmark.py')
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class BenchmarkValidationTests(unittest.TestCase):
    def setUp(self):
        self.bundle = copy.deepcopy(validator.load_bundle(ROOT))

    def test_current_bundle_is_valid(self):
        self.assertEqual(validator.validate(self.bundle)['citation_matches'], 50)

    def test_duplicate_id_is_rejected(self):
        self.bundle['records'][1]['requirement_id'] = 'NASA-SR-001'
        with self.assertRaisesRegex(ValueError, 'unique and ordered'):
            validator.validate(self.bundle, check_hashes=False)

    def test_wrong_clause_is_rejected(self):
        self.bundle['records'][0]['npr_clause'] = '9.9.9'
        with self.assertRaisesRegex(ValueError, 'clause does not match'):
            validator.validate(self.bundle, check_hashes=False)

    def test_unsupported_verdict_is_rejected(self):
        self.bundle['records'][0]['verdict'] = 'Compliant'
        with self.assertRaisesRegex(ValueError, 'invalid verdict'):
            validator.validate(self.bundle, check_hashes=False)

    def test_unrecorded_text_edit_is_rejected(self):
        self.bundle['records'][0]['requirement_text'] += ' Additional scope.'
        with self.assertRaisesRegex(ValueError, 'text-change flag'):
            validator.validate(self.bundle, check_hashes=False)

    def test_incomplete_reference_is_rejected(self):
        ref = next(r for r in self.bundle['clauses'] if r['swe_id'] == 'SWE-052')
        ref['clause_text'] = '3.12.1 The project manager shall maintain traceability. [SWE-052]'
        with self.assertRaisesRegex(ValueError, 'omitted required table'):
            validator.validate(self.bundle, check_hashes=False)

    def test_changed_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            folder = Path(path)
            for name in self.bundle['manifest']['files']:
                (folder / name).write_bytes((self.bundle['folder'] / name).read_bytes())
            with (folder / 'requirements.csv').open('a') as stream:
                stream.write('\n')
            self.bundle['folder'] = folder
            with self.assertRaisesRegex(ValueError, 'File hash mismatch'):
                validator.validate(self.bundle)


if __name__ == '__main__':
    unittest.main()
