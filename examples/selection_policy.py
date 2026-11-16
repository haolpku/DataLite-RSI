"""Example: prefer more output rows, optionally requiring a passed review."""


def select(context):
    candidates = context["candidates"]
    if context["config"].get("require_passed", False):
        candidates = [item for item in candidates if item["passed"]]
    measured = [item for item in candidates if item["output_rows"] is not None]
    if measured:
        best = max(
            measured,
            key=lambda item: (
                item["output_rows"], item["review_score"], item["iteration"]
            ),
        )
        reason = f"Largest measured output: {best['output_rows']} rows."
    else:
        previous = context["incumbent_iteration"]
        best = next((item for item in candidates if item["iteration"] == previous), None)
        if best is None:
            best = max(
                candidates,
                key=lambda item: (
                    item["passed"], item["review_score"], item["iteration"]
                ),
                default=None,
            )
        reason = "No row count available; retain the prior selection or use criteria."
    return {"iteration": best["iteration"] if best else None, "reason": reason}
