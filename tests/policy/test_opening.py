"""Opening phase routing and shop-conditioned planning contracts."""

from kaggriculture.policy.opening import controller, planner, post_shop_tape, pre_shop
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

    monkeypatch.setattr(controller.post_shop_tape, "action", unexpected)

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


def test_post_shop_phase_uses_the_tape_router(monkeypatch) -> None:
    expected = {"farmer": ["PASS"], "hands": [], "market": []}
    monkeypatch.setattr(post_shop_tape, "action", lambda observation, configuration=None: expected)

    assert controller.opening_action(make_fresh_observation(day=3)) is expected


def test_post_shop_tape_owns_days_three_through_eleven() -> None:
    assert not post_shop_tape.applies(2)
    assert post_shop_tape.applies(3)
    assert post_shop_tape.applies(11)
    assert not post_shop_tape.applies(12)


def test_bridge_tape_covers_days_three_through_five() -> None:
    from kaggriculture.policy.opening.bridge_tape import BRIDGE_TAPE

    assert len(BRIDGE_TAPE) == 72


def test_tape_bank_has_every_day3_day6_shop_pair() -> None:
    from kaggriculture.policy.opening.tape_bank import TAPE_BANK
    from kaggriculture.rules import constants as C

    expected_keys = {f"{a}|{b}" for a in C.SHOPS for b in C.SHOPS}
    assert set(TAPE_BANK) == expected_keys
    assert all(len(tape) == 144 for tape in TAPE_BANK.values())


def test_post_shop_tape_locks_in_at_day_six_and_stays_locked() -> None:
    from kaggriculture.policy.opening.tape_bank import lookup

    obs = make_fresh_observation(day=6)
    obs["step"] = 6 * 24
    obs["hour"] = 0
    obs["town"]["unlocked_shops"] = ["BAKERY", "YARN_STORE"]
    obs["farms"][0]["money"] = 0  # keep the reactive land overlay from firing here

    first = post_shop_tape.action(obs)
    expected = lookup("BAKERY", "YARN_STORE")[0]
    assert first["market"] == expected["market"]

    # Shops "changing" afterward (a stale observation, or the opponent's town
    # state bleeding in) must not re-route: the tape is locked for the rest
    # of the opening once selected.
    obs["step"] = 7 * 24
    obs["day"] = 7
    obs["town"]["unlocked_shops"] = ["PET_CAFE", "PET_CAFE"]
    second = post_shop_tape.action(obs)
    expected_next = lookup("BAKERY", "YARN_STORE")[24]
    assert second["market"] == expected_next["market"]


def test_tape_repair_forces_feeding_before_an_animal_escapes() -> None:
    from kaggriculture.policy.opening import tape_repair as repair

    obs = make_fresh_observation(day=6)
    farm = obs["farms"][0]
    x, y = farm["farmer"]
    farm["tiles"][y][x] = {
        "kind": "PASTURE",
        "animal": "COW",
        "fed_today": False,
        "consecutive_unfed": 1,
    }
    obs["private"]["inventories"][0] = {"WHEAT": 1}

    action = {"farmer": ["NORTH"], "hands": [], "market": []}
    repair.ensure_feeding(action, obs)

    assert action["farmer"] == ["FEED"]


def test_tape_repair_does_not_feed_without_wheat_anywhere() -> None:
    """No wheat in hand or in the shed: the rescue cannot conjure a feed
    out of nowhere, so it must never emit a bare FEED (which would silently
    no-op the way the tape's own drift already does)."""
    from kaggriculture.policy.opening import tape_repair as repair

    obs = make_fresh_observation(day=6)
    farm = obs["farms"][0]
    x, y = farm["farmer"]
    farm["tiles"][y][x] = {
        "kind": "PASTURE",
        "animal": "COW",
        "fed_today": False,
        "consecutive_unfed": 1,
    }

    action = {"farmer": ["NORTH"], "hands": [], "market": []}
    repair.ensure_feeding(action, obs)

    assert action["farmer"] != ["FEED"]


def test_tape_repair_routes_a_hand_to_the_shed_then_the_animal() -> None:
    """A hand with no assigned tape duty this turn (just moving) should be
    preempted to fetch wheat and walk toward an animal that a mismatched
    hand count left completely unattended, rather than leaving it stranded
    until it escapes."""
    from kaggriculture.policy.opening import tape_repair as repair

    obs = make_fresh_observation(day=6)
    farm = obs["farms"][0]
    farm["hands"] = [[0, 0]]
    obs["private"]["inventories"].append({})
    obs["private"]["shed"]["WHEAT"] = 5
    farm["tiles"][4][4] = {
        "kind": "PASTURE",
        "animal": "COW",
        "fed_today": False,
        "consecutive_unfed": 1,
    }

    action = {"farmer": ["PASS"], "hands": [["NORTH"]], "market": []}
    repair.ensure_feeding(action, obs)

    # Neither unit holds wheat yet, so the nearest one heads for the shed
    # rather than emitting a no-op FEED.
    assert action["farmer"] != ["FEED"]
    assert action["hands"][0] != ["FEED"]
    assert action["farmer"] != ["PASS"] or action["hands"][0] != ["NORTH"]


def test_post_shop_tape_forces_a_missed_land_purchase() -> None:
    """A tape's own BUY_LAND timing assumes its source game's cash. If the
    live game's cash trajectory diverges (e.g. the bridge and the locked-in
    tape come from different real games), the tape's single scripted
    attempt can be unaffordable and never retries on its own -- so a
    reactive overlay must force the purchase once cash allows it."""
    obs = make_fresh_observation(day=8)
    obs["step"] = 8 * 24
    obs["hour"] = 0
    obs["town"]["unlocked_shops"] = ["BAKERY", "YARN_STORE"]
    obs["farms"][0]["money"] = 10_000

    result = post_shop_tape.action(obs)

    assert ["BUY_LAND"] in result["market"]


def test_yarn_store_increases_sheep_target() -> None:
    from kaggriculture.policy.opening.targets import for_state

    # Replay evidence: without YARN_STORE, SHEEP stays flat at 3 for the
    # whole opening; the boost only shows up once shops unlock around day 6.
    base = for_state(6, [])
    yarn = for_state(6, ["YARN_STORE"])

    assert yarn.animals["SHEEP"] > base.animals["SHEEP"]


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
    obs = make_fresh_observation(day=5)
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
