"""CPU-only checks for the reproduction's data-integrity gates.

Run from this folder:
    py -X utf8 -m unittest discover -p 'test_repro_helpers.py' -v
Or select a separately cloned official source repository:
    py -X utf8 test_repro_helpers.py --repo /path/to/M3Diff -v
The M3DIFF_REPO environment variable also selects the source repository.
No model, GPU, image download, Java or metric package is needed.
"""
import argparse
import copy
import os
from pathlib import Path
import sys
import unittest

from repro_helpers import (
    COUNTS,
    postprocess_caption,
    read_json,
    validate_predictions,
    validate_questions,
)

DEFAULT_REPO = Path(__file__).resolve().parents[2] / "work" / "M3Diff"
REPO = Path(os.environ.get("M3DIFF_REPO", str(DEFAULT_REPO))).expanduser().resolve()


class PredictionIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.expected = ["real_test_pair_000001", "real_test_pair_000002"]
        self.rows = [
            {"image_id": self.expected[1], "caption": "The red chair disappeared."},
            {"image_id": self.expected[0], "caption": "A blue cup appeared on the table."},
        ]

    def test_valid_predictions_can_be_out_of_order(self):
        before = copy.deepcopy(self.rows)
        self.assertTrue(validate_predictions(self.rows, self.expected))
        self.assertEqual(self.rows, before)

    def test_missing_prediction_is_refused(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_predictions(self.rows[:1], self.expected)

    def test_extra_prediction_is_refused(self):
        rows = self.rows + [{"image_id": "unexpected_pair", "caption": "A box moved."}]
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_predictions(rows, self.expected)

    def test_equal_count_with_wrong_id_is_refused(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["image_id"] = "unexpected_pair"
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_predictions(rows, self.expected)

    def test_duplicate_prediction_is_refused(self):
        # Same row count as expected: the duplicate must not hide a missing pair.
        rows = [copy.deepcopy(self.rows[0]), copy.deepcopy(self.rows[0])]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_predictions(rows, self.expected)

    def test_empty_or_nonstring_caption_is_refused(self):
        for caption in ("", " \t\n ", None, 7, [], {}):
            with self.subTest(caption=caption):
                rows = copy.deepcopy(self.rows)
                rows[0]["caption"] = caption
                with self.assertRaisesRegex(ValueError, "caption"):
                    validate_predictions(rows, self.expected)

    def test_missing_caption_is_refused(self):
        rows = copy.deepcopy(self.rows)
        del rows[0]["caption"]
        with self.assertRaisesRegex(ValueError, "caption"):
            validate_predictions(rows, self.expected)

    def test_empty_file_is_not_full_test_success(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_predictions([], self.expected)

    def test_postprocess_caption_removes_loops_and_fixes_period_spacing(self):
        raw = (
            "The scene has become brighter.The pillow moved left. "
            "The cushion disappeared. The cushion disappeared. "
            "The cushion disappeared. Truncated tail without period"
        )
        cleaned = postprocess_caption(raw)
        self.assertEqual(
            cleaned,
            "The scene has become brighter. The pillow moved left. The cushion disappeared.",
        )


class OfficialEvaluationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = {}
        for split in ("Real", "Render"):
            base = REPO / "data" / "evaluation"
            question_path = base / "prompt" / f"OmniDiff_{split}_test.json"
            reference_path = base / "reference" / f"OmniDiff_{split}.json"
            if not question_path.is_file() or not reference_path.is_file():
                raise unittest.SkipTest(
                    f"Official evaluation files unavailable at {REPO}; "
                    "set M3DIFF_REPO or use --repo to enable these checks"
                )
            cls.data[split] = (read_json(question_path), read_json(reference_path))

    def test_official_counts_and_all_id_sets_match(self):
        expected_counts = {"Real": 803, "Render": 767}
        self.assertEqual(COUNTS, expected_counts)
        for split, expected_count in expected_counts.items():
            with self.subTest(split=split):
                questions, reference = self.data[split]
                question_ids = [q["id"] for q in questions]
                image_ids = [im["id"] for im in reference["images"]]
                annotation_ids = {a["image_id"] for a in reference["annotations"]}
                self.assertEqual(len(questions), expected_count)
                self.assertEqual(len(image_ids), expected_count)
                self.assertEqual(len(set(question_ids)), expected_count)
                self.assertEqual(set(question_ids), set(image_ids))
                self.assertEqual(set(question_ids), annotation_ids)
                self.assertEqual(validate_questions(questions, reference, split), question_ids)

    def test_validation_preserves_before_after_image_order(self):
        for split, (original_questions, reference) in self.data.items():
            with self.subTest(split=split):
                questions = copy.deepcopy(original_questions)
                before = copy.deepcopy(questions)
                # Include a descending filename pair, so an accidental sort is observable.
                self.assertTrue(any(q["image"][0] > q["image"][1] for q in questions))
                validate_questions(questions, reference, split)
                self.assertEqual(questions, before)

    def test_truncated_test_questions_are_refused(self):
        for split, (questions, reference) in self.data.items():
            with self.subTest(split=split):
                with self.assertRaisesRegex(ValueError, "expected"):
                    validate_questions(questions[:-1], reference, split)

    def test_duplicate_question_ids_are_refused(self):
        questions, reference = copy.deepcopy(self.data["Real"])
        questions[1]["id"] = questions[0]["id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_questions(questions, reference, "Real")

    def test_misaligned_reference_annotation_is_refused(self):
        questions, reference = copy.deepcopy(self.data["Render"])
        reference["annotations"][0]["image_id"] = "not_an_official_test_pair"
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_questions(questions, reference, "Render")

    def test_single_image_prompt_is_refused(self):
        questions, reference = copy.deepcopy(self.data["Real"])
        questions[0]["image"] = questions[0]["image"][:1]
        with self.assertRaisesRegex(ValueError, "two-image"):
            validate_questions(questions, reference, "Real")

    def test_training_image_in_test_prompt_is_refused(self):
        questions, reference = copy.deepcopy(self.data["Render"])
        questions[0]["image"][0] = "Render/train/images/render_train_000001.png"
        with self.assertRaisesRegex(ValueError, "test image path"):
            validate_questions(questions, reference, "Render")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo", type=Path)
    options, unittest_args = parser.parse_known_args()
    if options.repo is not None:
        REPO = options.repo.expanduser().resolve()
    unittest.main(argv=[sys.argv[0], *unittest_args])
