import tempfile
import unittest
from pathlib import Path

from src.eval_harness import escalation_metrics, intent_metrics
from src.escalate import decide
from src.retrieve import Playbook


class EscalationTests(unittest.TestCase):
    def test_sensitive_keyword_escalates(self):
        result = decide(
            "I am filing a chargeback today",
            {"intent": "billing_subscription", "confidence": 0.9},
            {"grounded": True},
        )
        self.assertTrue(result["escalate"])
        self.assertTrue(any("chargeback" in reason for reason in result["reasons"]))

    def test_safe_grounded_message_does_not_escalate(self):
        result = decide(
            "How do I restart my phone?",
            {"intent": "feature_howto", "confidence": 0.9},
            {"grounded": True},
        )
        self.assertFalse(result["escalate"])


class RetrievalTests(unittest.TestCase):
    def test_empty_query_has_no_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "playbook.csv"
            path.write_text(
                "customer_text,brand_reply,intent\n"
                "phone will not turn on,Try a restart,device_troubleshooting\n",
                encoding="utf-8",
            )
            self.assertEqual(Playbook(path).top_k("device_troubleshooting", ""), [])


class MetricsTests(unittest.TestCase):
    def test_metrics_counts_match_confusion_matrix(self):
        rows = [
            {"tweet_id": "1", "intent": "account_access", "should_escalate": "True"},
            {"tweet_id": "2", "intent": "other_general", "should_escalate": "False"},
        ]
        predictions = {
            "1": {"intent": "account_access", "escalate": True},
            "2": {"intent": "account_access", "escalate": False},
        }
        self.assertEqual(intent_metrics(rows, predictions)["accuracy"], 0.5)
        self.assertEqual(
            escalation_metrics(rows, predictions),
            {
                "tp": 1,
                "fp": 0,
                "fn": 0,
                "tn": 1,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "note": "fn (missed escalations) is the metric to watch most closely",
            },
        )


if __name__ == "__main__":
    unittest.main()
