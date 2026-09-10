"""Program labelers run on plain traces without a machine or model."""
from skill2fsm.schema import Record, Trace
from skill2fsm.trace_adapter import LABELERS, branch_label


def test_next_action_after_labels_the_following_step():
    records = [
        Record(step=1, action={"kind": "tool", "name": "read"}),
        Record(step=2, action={"kind": "tool", "name": "bash"}, output={"stdout": "done"}),
    ]
    trace = Trace(records=records)
    label = LABELERS["next_action_after"]
    assert label(trace, 0) == branch_label(records[1])
    assert label(trace, -1) == branch_label(records[0])
    assert label(trace, 1) is None
