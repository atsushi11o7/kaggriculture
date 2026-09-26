"""Opening phase routing and shop-conditioned planning contracts."""

from kaggriculture.policy.opening import controller, planner, pre_shop
from tests.policy.conftest import make_fresh_observation


def test_pre_shop_phase_owns_days_zero_through_two() -> None:
    assert pre_shop.applies(0)
    assert pre_shop.applies(2)
    assert not pre_shop.applies(3)


def test_pre_shop_action_skips_later_overrides(monkeypatch) -> None:
    expected = {"farmer": ["PASS"], "hands": [], "market": []}
    monkeypatch.setattr(pre_shop, "action", lambda observation, configuration=None: expected)

    def unexpected(*args, **kwargs):
        raise AssertionError("post-shop override ran during the pre-shop phase")

    monkeypatch.setattr(controller.planner, "action", unexpected)

    assert controller.opening_action(make_fresh_observation(day=2)) is expected


def test_pre_shop_is_independent_from_vendored_hybrid() -> None:
    import inspect

    from kaggriculture.policy.opening import pre_shop_tape

    assert len(pre_shop_tape.PRE_SHOP_TAPE) == 72
    assert "hybrid2965_agent" not in inspect.getsource(pre_shop)


def test_pre_shop_repairs_a_weed_blocking_a_planned_action() -> None:
    from copy import deepcopy

    from kaggriculture.policy.opening.pre_shop_tape import PRE_SHOP_TAPE

    step = next(
        index
        for index, planned in enumerate(PRE_SHOP_TAPE)
        if (planned.get("farmer") or ["PASS"])[0] in {"PLANT", "BUILD_COOP", "BUILD_PASTURE"}
    )
    obs = make_fresh_observation(day=step // 24)
    obs["step"] = step
    obs["hour"] = step % 24
    x, y = obs["farms"][0]["farmer"]
    obs["farms"][0]["tiles"][y][x] = {"kind": "WEED"}

    result = pre_shop.action(deepcopy(obs))

    assert result["farmer"] == ["DIG"]


def test_opening_hands_off_after_raw_day_eleven() -> None:
    assert controller.is_opening_day(11)
    assert not controller.is_opening_day(12)


def test_post_shop_phase_uses_state_driven_planner(monkeypatch) -> None:
    expected = {"farmer": ["PASS"], "hands": [], "market": []}
    monkeypatch.setattr(planner, "action", lambda observation, configuration=None: expected)

    assert controller.opening_action(make_fresh_observation(day=3)) is expected


def test_yarn_store_increases_sheep_target() -> None:
    from kaggriculture.policy.opening.targets import for_state

    base = for_state(3, [])
    yarn = for_state(3, ["YARN_STORE"])

    assert yarn.animals["SHEEP"] == base.animals["SHEEP"] + 7


def test_planner_matches_observed_hand_count() -> None:
    obs = make_fresh_observation(day=3)
    obs["farms"][0]["hands"] = [[4, 4], [5, 4]]
    obs["private"]["inventories"].extend([{}, {}])

    result = planner.action(obs)

    assert len(result["hands"]) == 2
    assert len(result["market"]) <= 10


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


def test_sales_can_fund_due_land_in_the_same_turn() -> None:
    obs = make_fresh_observation(day=6)
    obs["farms"][0]["money"] = 1_100
    obs["private"]["shed"]["CARROT"] = 3

    market = planner.action(obs)["market"]

    assert ["SELL", "CARROT", 3] in market
    assert ["BUY_LAND"] in market
    assert market.index(["SELL", "CARROT", 3]) < market.index(["BUY_LAND"])


def test_expansion_inputs_do_not_consume_reserved_land_cash() -> None:
    obs = make_fresh_observation(day=3)
    obs["farms"][0]["money"] = 900

    market = planner.action(obs)["market"]

    assert not any(order[0] in {"BUY_ANIMAL", "BUY_SEED"} for order in market)


def test_newly_unlocked_quadrant_receives_first_plant_tasks() -> None:
    from kaggriculture.policy.opening.targets import for_state

    obs = make_fresh_observation(day=11)
    farm = obs["farms"][0]
    farm["unlocked_quadrants"] = ["NW", "NE", "SW", "SE"]
    for row in farm["tiles"]:
        for x, tile in enumerate(row):
            if tile == "LOCKED":
                row[x] = None
    for x, y in ((0, 0), (5, 0), (0, 5)):
        farm["tiles"][y][x] = {"kind": "PASTURE"}
    obs["private"]["seeds"]["STRAWBERRY"] = 4

    tasks = planner._tasks(obs, for_state(10, []))
    plants = [task for task in tasks if task.command[0] == "PLANT"]

    assert plants
    first_x, first_y = plants[0].position
    assert first_x >= 5 and first_y >= 5
    assert {planner._quadrant(task.position) for task in plants[:4]} == {"NW", "NE", "SW", "SE"}
