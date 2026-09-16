"""context_probe: for states with model calls, report a comparison of output quality under narrow vs. wide reads."""

from hexis.examples import table_clean as tc
from hexis.legacy import compiler, report


def test_context_probe_compares_narrow_and_wide_context(accepted):
    cr = compiler.compile(tc.skill_doc(), accepted(24, seed=2), skill_id="tc",
                          model=tc.build_model(),
                          prohibitions=tc.reference_machine().prohibitions)
    probe = report.context_probe(
        cr.machine, tc.gen_tasks(10, seed=88),
        model=tc.build_model(),
        tools=lambda task: tc.build_registry(tc.MemFS(task["files"])),
        doc=tc.skill_doc(), wide_extra={"rows": None})

    judge_ids = [sid for sid, s in cr.machine.states.items()
                 if s.action.kind == "judge"]
    assert judge_ids
    for sid in judge_ids:
        d = probe[sid]
        assert d["samples"] > 0                        # judge inputs were really sampled from the task stream
        assert d["narrow_vs_wide_agree"] is not None   # the comparison of the two contexts is in the report
        assert 0.0 <= d["narrow_vs_wide_agree"] <= 1.0
        assert d["label_dist"]                         # the output distribution is there too


def test_cover_report_renders(accepted):
    traces = accepted(24, seed=2)
    cr = compiler.compile(tc.skill_doc(), traces, skill_id="tc",
                          prohibitions=tc.reference_machine().prohibitions)
    rep = report.cover_report(cr.machine, compiler.partition(tc.skill_doc()),
                              t_plus=traces)
    text = report.render(rep)
    assert "coverage report" in text and "S2.1" in text
