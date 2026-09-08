import pytest

from app.degradation import DegradationPolicy


@pytest.mark.parametrize(
    ("level", "delay"), [(0, 0), (25, 0), (50, 2000 / 3), (75, 4000 / 3), (100, 2000)]
)
def test_proportional_delay(level, delay):
    assert DegradationPolicy().decide(level).delay_ms == pytest.approx(delay)


def test_monotonic_delay():
    policy = DegradationPolicy()
    delays = [policy.decide(level).delay_ms for level in range(101)]
    assert delays == sorted(delays)


def test_repeatable_sequence():
    levels = [0, 25, 50, 75, 100] * 20
    first, second = DegradationPolicy(seed=42), DegradationPolicy(seed=42)
    assert [first.decide(level) for level in levels] == [
        second.decide(level) for level in levels
    ]


def test_seed_changes_sequence():
    first, second = DegradationPolicy(seed=42), DegradationPolicy(seed=43)
    assert [first.decide(100) for _ in range(20)] != [
        second.decide(100) for _ in range(20)
    ]


@pytest.mark.parametrize(("seed", "status"), [(42, 500), (1, 503)])
def test_controlled_failures(seed, status):
    decision = DegradationPolicy(seed=seed).decide(100)
    assert decision.status_code == status
    assert decision.degradation_type == f"error_{status}"


def test_policy_boundaries():
    assert DegradationPolicy().decide(25).degradation_type == "none"
    assert DegradationPolicy().decide(26).degradation_type == "delay"
    assert DegradationPolicy(seed=1).decide(74).status_code is None
    assert DegradationPolicy(seed=1).decide(75).status_code == 500
    assert DegradationPolicy(seed=1).decide(90).status_code == 500
    assert DegradationPolicy(seed=31).decide(91).status_code == 503


def test_configurable_max_delay():
    assert DegradationPolicy(max_delay_ms=300).decide(100).delay_ms == 300
    assert DegradationPolicy(max_delay_ms=0).decide(50).degradation_type == "none"
    with pytest.raises(ValueError):
        DegradationPolicy(max_delay_ms=-1)


@pytest.mark.parametrize("level", [-1, 101, True, 50.5, "50"])
def test_invalid_level(level):
    with pytest.raises(ValueError):
        DegradationPolicy().decide(level)
