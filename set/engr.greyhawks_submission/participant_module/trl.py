"""TRL decision from criterion results (pure functions, easy to test)."""

from __future__ import annotations


def decide_trl(status: dict, rung: dict, gap: int = 2):
    """Return (trl, abstention_reason, rationale).

    Start at the highest TRL rung whose criterion is `met`. Believe it only when the ladder below it is
    supported: rung <= 2, or one of the two rungs directly below is also met. Otherwise step down to the
    next met rung. Nothing supported -> abstain. (Your branch used max(met) which lets one stray
    high-rung match inflate the TRL; an error of 2 or more levels scores 0.)
    """
    met = sorted({rung[c] for c, s in status.items() if s == "met" and c in rung})
    if not met:
        return None, "insufficient evidence", "No TRL-rung criterion has supporting evidence."
    for candidate in reversed(met):
        if candidate <= 2 or any((candidate - k) in met for k in range(1, gap + 1)):
            note = "" if candidate == met[-1] else f" (rung {met[-1]} ignored: the rungs below it are not supported)"
            return candidate, None, f"Highest rung with a supported ladder is TRL {candidate}{note}."
    return None, "conflicting evidence", f"Only isolated rungs {met} are met; the ladder is not supported."


def a_score(predicted, estimate) -> float:
    """Part-A score used for local evaluation: exact 1, off by 1 0.5, abstain 0.25, otherwise 0."""
    if predicted is None:
        return 0.25
    d = abs(predicted - estimate)
    return 1.0 if d == 0 else 0.5 if d == 1 else 0.0


def decide_from_proba(proba: dict, abstain_below: float = 0.25):
    """proba {trl: prob}. Pick the TRL with the best expected part-A score (exact 1, off-by-1 0.5);
    abstain (None) when even that is worth less than the 0.25 an abstention earns."""
    exp = {t: proba.get(t, 0.0) + 0.5 * (proba.get(t - 1, 0.0) + proba.get(t + 1, 0.0)) for t in range(1, 10)}
    best = max(exp, key=exp.get)
    return (best, exp[best]) if exp[best] >= abstain_below else (None, exp[best])
