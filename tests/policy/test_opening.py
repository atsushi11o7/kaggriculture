"""Reactive land-purchase safety net used by the submission agent."""

from tests.policy.conftest import make_fresh_observation


def test_land_purchase_windows_match_raw_observation_days() -> None:
    from kaggriculture.policy.opening.land_rule import next_forced_purchase

    obs = make_fresh_observation(day=5)
    obs["farms"][0]["money"] = 10_000
    assert next_forced_purchase(obs) is None

    obs["day"] = 6
    assert next_forced_purchase(obs) == ["BUY_LAND"]

    obs["farms"][0]["unlocked_quadrants"].append("NE")
    obs["day"] = 8
    assert next_forced_purchase(obs) is None
    obs["day"] = 9
    assert next_forced_purchase(obs) == ["BUY_LAND"]

    obs["farms"][0]["unlocked_quadrants"].append("SW")
    obs["day"] = 10
    assert next_forced_purchase(obs) == ["BUY_LAND"]
