"""Counterfactual resource swap.

Two contracts:
1. Returned dict has actual_total_h ≈ sum of per-event actual_wait_h.
2. With swap='alice=bob', any event whose actual resource is 'alice'
   has its cf_resource set to 'bob'; others keep the original resource.
"""

from __future__ import annotations


def test_whatif_totals_match_per_event_sum(synthetic_event_log):
    from gnn_cli.whatif import predict_whatif
    from modules.data_preprocessing import encode_categoricals

    df, le_task, _ = encode_categoricals(synthetic_event_log)
    case = df["case_id"].iloc[0]
    result = predict_whatif(df, case, ("alice", "bob"), le_task=le_task)
    assert result["case_id"] == case
    actual_sum = sum(r["actual_wait_h"] for r in result["per_event"])
    cf_sum = sum(r["cf_wait_h"] for r in result["per_event"])
    assert abs(result["actual_total_h"] - actual_sum) < 1e-6
    assert abs(result["counterfactual_total_h"] - cf_sum) < 1e-6
    assert abs(result["delta_total_h"] - (cf_sum - actual_sum)) < 1e-6


def test_whatif_only_swaps_matching_resource(synthetic_event_log):
    from gnn_cli.whatif import predict_whatif
    from modules.data_preprocessing import encode_categoricals

    df, le_task, _ = encode_categoricals(synthetic_event_log)
    case = df["case_id"].iloc[0]
    result = predict_whatif(df, case, ("alice", "bob"), le_task=le_task)
    for r in result["per_event"]:
        if r["actual_resource"] == "alice":
            assert r["cf_resource"] == "bob"
        else:
            assert r["cf_resource"] == r["actual_resource"]


def test_whatif_rejects_unknown_case(synthetic_event_log):
    import pytest
    from gnn_cli.whatif import predict_whatif
    from modules.data_preprocessing import encode_categoricals

    df, _, _ = encode_categoricals(synthetic_event_log)
    with pytest.raises(ValueError, match="not found"):
        predict_whatif(df, "no_such_case", ("alice", "bob"))
