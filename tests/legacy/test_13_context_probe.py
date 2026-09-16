"""⑬ context_probe：对含模型调用的状态输出窄读/宽读的产出质量对比报告。"""

from hexis.legacy import compiler, report
from hexis.examples import table_clean as tc


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
        assert d["samples"] > 0                        # 真在任务流里采到了判断输入
        assert d["narrow_vs_wide_agree"] is not None   # 两种上下文的对比在报告里
        assert 0.0 <= d["narrow_vs_wide_agree"] <= 1.0
        assert d["label_dist"]                         # 产出分布也在


def test_cover_report_renders(accepted):
    traces = accepted(24, seed=2)
    cr = compiler.compile(tc.skill_doc(), traces, skill_id="tc",
                          prohibitions=tc.reference_machine().prohibitions)
    rep = report.cover_report(cr.machine, compiler.partition(tc.skill_doc()),
                              t_plus=traces)
    text = report.render(rep)
    assert "覆盖报告" in text and "S2.1" in text
