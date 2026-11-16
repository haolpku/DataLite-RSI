import unittest

from video_rsi.operators.production import DistractorQualityOperator


class DistractorQualityTests(unittest.TestCase):
    def setUp(self):
        self.operator = DistractorQualityOperator()

    def _sample(self, **updates):
        sample = {
            "question": "What does the person do after entering the room?",
            "answer": "The person places the red cup on the table",
            "choices": [
                "The person places the red cup on the table",
                "The person picks up the blue book from the chair",
                "The person opens the window beside the table",
                "The person leaves the room through the doorway",
            ],
        }
        sample.update(updates)
        return sample

    def test_accepts_balanced_hard_negatives(self):
        result = self.operator.run({"enhanced_samples": [self._sample()]}, None)
        self.assertEqual(1, len(result["distractor_checked_samples"]))
        self.assertEqual(1, result["distractor_feedback"]["accepted_samples"])

    def test_rejects_generic_or_duplicate_distractor(self):
        sample = self._sample(choices=[
            "The person places the red cup on the table",
            "Cannot determine",
            "The person places the red cup on the table",
            "The person opens the window beside the table",
        ])
        result = self.operator.run({"enhanced_samples": [sample]}, None)
        self.assertEqual([], result["distractor_checked_samples"])
        self.assertTrue(any(key.startswith("rejected_") for key in result["distractor_feedback"]))

    def test_rejects_answer_leakage(self):
        sample = self._sample(question="Does the person place the red cup on the table?")
        result = self.operator.run({"enhanced_samples": [sample]}, None)
        self.assertEqual([], result["distractor_checked_samples"])
        self.assertEqual(1, result["distractor_feedback"]["rejected_answer_leaked_in_question"])

    def test_allows_lexically_similar_but_visually_distinct_negative(self):
        sample = self._sample(choices=[
            "The person places the red cup on the table",
            "The person places the red cup beside the table",
            "The person picks up the blue book from the chair",
            "The person leaves the room through the doorway",
        ])
        result = self.operator.run({"enhanced_samples": [sample]}, None)
        self.assertEqual(1, len(result["distractor_checked_samples"]))
        self.assertNotIn(
            "rejected_distractor_near_duplicate_of_answer",
            result["distractor_feedback"],
        )

    def test_enhancement_preserves_answer(self):
        from video_rsi.operators.production import DistractorEnhancementOperator

        class FakeEnhancer:
            name = "fake_enhancer"

            def enhance(self, *, sample, evidence, prompt):
                return {"choices": [
                    sample["answer"],
                    "The person lifts the blue cup from the table",
                    "The person opens the nearby window",
                    "The person walks toward the doorway",
                ]}

        sample = self._sample()
        result = DistractorEnhancementOperator(FakeEnhancer()).run(
            {"generated_samples": [sample]}, None
        )
        self.assertEqual("enhanced", result["enhanced_samples"][0]["distractor_enhancement"]["status"])
        self.assertEqual(4, len(result["enhanced_samples"][0]["choices"]))


if __name__ == "__main__":
    unittest.main()
