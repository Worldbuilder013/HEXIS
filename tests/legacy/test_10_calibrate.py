"""Judge action error rates are calibrated and written into the machine, with examples taken from traces."""

from hexis.examples import table_clean as tc
from hexis.legacy import compiler
from hexis.machine.schema import JudgeAction


def test_calibrate_recovers_injected_error_rate():
    """Given well-formed header snapshots with correct labels, a model with 20% injected error should calibrate to about 0.2."""
    judge = JudgeAction(prompt=tc.JUDGE_Q, reads=["header_row"],
                        writes=["header_ok"], labels=["well_formed", "malformed", "abstain"])
    # 50 distinct well-formed headers (different fingerprints -> error injection spreads proportionally)
    labeled = [({"header_row": f"name{i},quantity,date"}, "well_formed") for i in range(50)]
    noisy = tc.build_model(error_rate=0.2, seed=1)
    rate, support = compiler.calibrate(judge, labeled, noisy)
    assert support == 50
    assert 0.08 <= rate <= 0.34, rate                 # around the injected 0.2


def test_perfect_model_calibrates_to_zero():
    judge = JudgeAction(prompt=tc.JUDGE_Q, reads=["header_row"],
                        writes=["header_ok"], labels=["well_formed", "malformed", "abstain"])
    labeled = [({"header_row": f"name{i},quantity"}, "well_formed") for i in range(30)]
    rate, _ = compiler.calibrate(judge, labeled, tc.build_model())
    assert rate == 0.0


def test_compiled_judge_carries_examples_and_error_rate(accepted):
    cr = compiler.compile(tc.skill_doc(), accepted(24, seed=2),
                          skill_id="table-clean", model=tc.build_model(),
                          prohibitions=tc.reference_machine().prohibitions)
    judge_state = next(s for s in cr.machine.states.values()
                       if s.action.kind == "judge")
    assert judge_state.action.examples, "the judge action should carry examples taken from traces"
    # example fields are exactly the reads, and each label is in the observed label set
    for ex in judge_state.action.examples:
        d = ex.model_dump()
        assert "header_row" in d and d["label"] in judge_state.action.labels
    sid = judge_state.id
    assert sid in cr.calibration
    assert cr.calibration[sid]["support"] > 0


def test_task_inputs_never_become_branch_predicates():
    """A task input is the identity of "which problem this is", not something that happened during
    execution. A branch guard learned from it inevitably says "if the input workbook is
    /tmp/<sandbox>/input.xlsx, go this way": 100% separable on the training set, and forever false
    on another run. This has actually happened in real compiled machines."""
    from hexis.legacy.fit import candidate_atoms, learn_cond
    from hexis.machine.schema import Variable

    variables = [
        Variable(name="input_path", type="string", init_from="task.input.input_path"),
        Variable(name="request", type="string", init_from="task.input.request"),
        Variable(name="returncode", type="integer", init=0),
    ]
    snaps = {
        "s_ok":  [{"input_path": "/tmp/a/input.xlsx", "request": "problem 1", "returncode": 0},
                  {"input_path": "/tmp/b/input.xlsx", "request": "problem 2", "returncode": 0}],
        "s_bad": [{"input_path": "/tmp/c/input.xlsx", "request": "problem 3", "returncode": 1},
                  {"input_path": "/tmp/d/input.xlsx", "request": "problem 4", "returncode": 2}],
    }
    atoms = candidate_atoms(snaps, variables)
    assert not any("input_path" in a or "request" in a for a in atoms), atoms
    assert any("returncode" in a for a in atoms)

    learned = learn_cond(snaps, variables, min_support=2, holdout_ratio=0.0, acc_thr=0.9)
    assert learned is not None
    for cond in learned.values():
        assert "input_path" not in cond and "request" not in cond
        assert "/tmp/" not in cond
