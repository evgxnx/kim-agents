"""Comparing answer values: `12`, `12.0` and `12,0` are the same answer; names compare case-insensitively."""


def _number(value: str) -> float | None:
    try:
        return float(value.strip().rstrip("%").replace(",", ".").replace(" ", ""))
    except ValueError:
        return None


def same_value(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    na, nb = _number(a), _number(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= max(1e-6, 1e-3 * abs(nb))
    return a.strip().casefold() == b.strip().casefold()
