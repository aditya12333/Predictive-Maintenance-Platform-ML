import json

from predictive_maintenance.monitoring.retraining import evaluate_report_history
from predictive_maintenance.monitoring.retraining_policy import RetrainingDecision


def test_report_history_requests_candidate_after_consecutive_failures(tmp_path) -> None:
    for index in (1, 2):
        (tmp_path / f"20260101T000{index}00Z.json").write_text(
            json.dumps(
                {
                    "checks": [
                        {
                            "name": "model_mae_degradation",
                            "value": 30.0,
                            "status": "fail",
                        }
                    ]
                }
            )
        )

    result = evaluate_report_history(tmp_path)

    assert result.decision is RetrainingDecision.REQUEST_CANDIDATE
