"""⑩ 判断动作误差率被标定并写入机器，样例来自轨迹。"""

from hexis.legacy import compiler
from hexis.examples import table_clean as tc
from hexis.machine.schema import JudgeAction


def test_calibrate_recovers_injected_error_rate():
    """给一批带正确标签的规范表头快照，注入 20% 误差的模型应被标定出 ~0.2。"""
    judge = JudgeAction(prompt=tc.JUDGE_Q, reads=["header_row"],
                        writes=["header_ok"], labels=["规范", "不规范", "弃权"])
    # 50 个各不相同的规范表头（不同指纹 → 误差注入按比例铺开）
    labeled = [({"header_row": f"名称{i},数量,日期"}, "规范") for i in range(50)]
    noisy = tc.build_model(error_rate=0.2, seed=1)
    rate, support = compiler.calibrate(judge, labeled, noisy)
    assert support == 50
    assert 0.08 <= rate <= 0.34, rate                 # 围绕注入的 0.2


def test_perfect_model_calibrates_to_zero():
    judge = JudgeAction(prompt=tc.JUDGE_Q, reads=["header_row"],
                        writes=["header_ok"], labels=["规范", "不规范", "弃权"])
    labeled = [({"header_row": f"名称{i},数量"}, "规范") for i in range(30)]
    rate, _ = compiler.calibrate(judge, labeled, tc.build_model())
    assert rate == 0.0


def test_compiled_judge_carries_examples_and_error_rate(accepted):
    cr = compiler.compile(tc.skill_doc(), accepted(24, seed=2),
                          skill_id="table-clean", model=tc.build_model(),
                          prohibitions=tc.reference_machine().prohibitions)
    judge_state = next(s for s in cr.machine.states.values()
                       if s.action.kind == "judge")
    assert judge_state.action.examples, "判断动作应带取自轨迹的样例"
    # 样例的字段就是 reads，标签在观测标签集里
    for ex in judge_state.action.examples:
        d = ex.model_dump()
        assert "header_row" in d and d["label"] in judge_state.action.labels
    sid = judge_state.id
    assert sid in cr.calibration
    assert cr.calibration[sid]["support"] > 0


def test_task_inputs_never_become_branch_predicates():
    """任务输入是「这次是哪道题」的身份，不是执行中发生的事。拿它当分岔条件学出来的
    必然是「如果输入簿是 /var/folders/…/input.xlsx 就走这边」——训练集上百分百可分，
    换一次运行就永远为假。实测产物里真出现过（out/skeleton/xlsx_new/machine_41691.json
    的 s26）。"""
    from hexis.legacy.fit import candidate_atoms, learn_cond
    from hexis.machine.schema import Variable

    variables = [
        Variable(name="input_path", type="string", init_from="task.input.input_path"),
        Variable(name="request", type="string", init_from="task.input.request"),
        Variable(name="returncode", type="integer", init=0),
    ]
    snaps = {
        "s_ok":  [{"input_path": "/tmp/a/input.xlsx", "request": "题一", "returncode": 0},
                  {"input_path": "/tmp/b/input.xlsx", "request": "题二", "returncode": 0}],
        "s_bad": [{"input_path": "/tmp/c/input.xlsx", "request": "题三", "returncode": 1},
                  {"input_path": "/tmp/d/input.xlsx", "request": "题四", "returncode": 2}],
    }
    atoms = candidate_atoms(snaps, variables)
    assert not any("input_path" in a or "request" in a for a in atoms), atoms
    assert any("returncode" in a for a in atoms)

    learned = learn_cond(snaps, variables, min_support=2, holdout_ratio=0.0, acc_thr=0.9)
    assert learned is not None
    for cond in learned.values():
        assert "input_path" not in cond and "request" not in cond
        assert "/tmp/" not in cond
