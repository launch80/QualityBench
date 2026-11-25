from ..taskdef import Task

TASK = Task(
    id="lcb", label="LiveCodeBench v6", items=175, order=70, cost=17,
    how="fixed 175-problem slice, sandboxed",
    about="Competition programming problems (LiveCodeBench) from LeetCode, AtCoder and Codeforces, collected after most models' training data. Solutions are run against hidden tests.",
    scored=True, resumable=True, ref_minutes=17, overhead_s=35, needs=frozenset({"lcb"}),
)
