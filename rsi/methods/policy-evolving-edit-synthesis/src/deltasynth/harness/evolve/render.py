from __future__ import annotations

from .state import StrategyState


def strategy_ref(state: StrategyState) -> str:
    return f"strategy@{state.version}"


def render_directives(state: StrategyState) -> list[str]:
    out: list[str] = []
    if state.rendered_prompt:
        out.append(state.rendered_prompt)
    else:
        out.extend(
            line for line in (p.render() for p in state.active_policies()) if line
        )
    if state.diversity_categories:
        counts = ", ".join(
            f"{name}: {count}"
            for name, count in sorted(
                state.diversity_categories.items(), key=lambda kv: -kv[1]
            )
        )
        out.append(f"Current run counts: {counts}.")
    return out


def render_memory(state: StrategyState) -> str:
    out = [
        f"# strategy@{state.version}  ({state.edit_code}, from batch "
        f"{state.observed_batch})"
    ]
    if state.diagnosis:
        out.append(f"\n## diagnosis\n{state.diagnosis}")
    if state.rationale:
        out.append(f"\n## rationale\n{state.rationale}")

    active = state.active_policies()
    out.append(f"\n## active policies ({len(active)})")
    for policy in active:
        out.append(
            f"\n### {policy.policy_id} (v{policy.version})\n"
            f"- in force since {policy.deployed_at_batch}, "
            f"from claim `{policy.source_hypothesis}`\n"
            f"- when: {policy.trigger}\n"
            f"- do: {policy.procedure}\n"
            f"- except: {policy.boundary or '(none stated)'}"
        )

    dropped = [p for p in state.policies if not p.active]
    if dropped:
        out.append(f"\n## withdrawn policies ({len(dropped)})")
        for policy in dropped:
            out.append(
                f"- {policy.policy_id}: {policy.removed_reason or 'never promoted'}"
            )

    out.append(f"\n## claims ({len(state.hypotheses)})")
    for hypothesis in state.hypotheses:
        line = (
            f"- [{hypothesis.status}] {hypothesis.hypothesis_id}: "
            f"supported in {hypothesis.support}, contradicted in "
            f"{hypothesis.oppose}, untested streak {hypothesis.untested_streak}"
        )
        if hypothesis.retired_reason:
            line += f" — retired: {hypothesis.retired_reason}"
        out.append(line)
        out.append(f"  {hypothesis.statement}")
    return "\n".join(out)
