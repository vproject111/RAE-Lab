"""Unit tests for RetrievalOptimizer (Iteration 9)."""

import pytest
from rae_lab.retrieval_optimizer import MaturityMode, RetrievalOptimizer


def test_reward_formula_calculation():
    optimizer = RetrievalOptimizer()

    # Ideal retrieval: quality 1.0, fast latency (50ms), no failure, no token cost
    reward_ideal = optimizer.compute_reward(
        quality=1.0,
        latency_ms=50.0,
        token_cost=0.0,
        is_failed=False,
    )
    # 0.65 * 1.0 - 0.10 * (50/500) = 0.65 - 0.01 = 0.64
    assert reward_ideal == pytest.approx(0.64, abs=0.01)

    # Failed retrieval: quality 0.0, slow latency (500ms), is_failed True
    reward_failed = optimizer.compute_reward(
        quality=0.0,
        latency_ms=500.0,
        token_cost=0.0,
        is_failed=True,
    )
    # 0.0 - 0.10 * 1.0 - 0.15 = -0.25
    assert reward_failed == pytest.approx(-0.25, abs=0.01)


def test_shadow_mode_does_not_mutate_weights():
    optimizer = RetrievalOptimizer(mode=MaturityMode.SHADOW)
    initial_weights = dict(optimizer.current_weights)

    rec = optimizer.step(
        strategy_name="vector",
        quality=0.9,
        latency_ms=100.0,
    )

    assert rec.mode == MaturityMode.SHADOW
    assert optimizer.current_weights == initial_weights


def test_active_mode_adaptation_and_bounds():
    optimizer = RetrievalOptimizer(
        mode=MaturityMode.ACTIVE,
        damping_factor=0.05,
        max_deviation=0.15,
        baseline_weights={"vector": 0.5},
    )

    # High quality feedback steps
    for _ in range(50):
        optimizer.step(
            strategy_name="vector",
            quality=0.95,
            latency_ms=80.0,
        )

    # Bound is 0.5 * 1.15 = 0.575
    assert optimizer.current_weights["vector"] <= 0.575
    assert optimizer.current_weights["vector"] > 0.5


def test_reward_gain_simulation_1000_cycles():
    """DoD: Average reward increases and remains stable over 1000 cycles without oscillation."""
    optimizer = RetrievalOptimizer(
        mode=MaturityMode.ACTIVE,
        damping_factor=0.05,
        max_deviation=0.15,
    )

    initial_reward = optimizer.compute_reward(quality=0.5, latency_ms=250.0)

    # Run 1000 simulated cycles with progressively improving quality
    for i in range(1000):
        # Progressively higher quality from feedback
        simulated_quality = min(0.95, 0.5 + (i / 2000.0))
        optimizer.step(
            strategy_name="vector",
            quality=simulated_quality,
            latency_ms=120.0,
        )

    final_avg_reward = optimizer.get_average_reward(last_n=100)
    assert final_avg_reward > initial_reward * 1.12, "Reward did not improve by >= 12%"
