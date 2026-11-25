from ..taskdef import Task

TASK = Task(
    id="bfcl", label="BFCL", items=400, order=60, cost=4,
    how="4 categories, function calling via the server tool parser",
    about="Function calling (Berkeley Function Calling Leaderboard). Given tool definitions, the model has to choose the right function and fill in its arguments correctly, including cases with several calls or no fitting tool.",
    scored=True, ref_minutes=4, overhead_s=5, needs=frozenset({"bfcl"}),
)
