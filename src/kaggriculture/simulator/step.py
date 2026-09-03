"""1ターン分の状態遷移(元コードのinterpreter。初期化(_initialize)はreset.pyが担う)。

処理順序は元コードのinterpreterと同じ:
1. ユニット行動(両プレイヤー、farmer→hands、PLANTのブロック判定込み)
2. 市場注文キュー(両プレイヤー)
3. 街の消費
4. 減衰
5. (日末のみ)日次更新・雑草発生・店の抽選・持ち物の払い出し・位置リセット

日末処理は「毎ターン計算し、日末でなければjnp.whereで結果を捨てる」方式にしている
(unit_actions.py等と同じ合成パターン)。このコストを日末以外のターンで削りたい
場合はstep_batch_lockstep()を使う(条件・トレードオフはそちらのdocstring参照)。
"""

import functools

import jax
import jax.numpy as jnp

from . import random_events
from .action import Action
from .daily_refresh import apply_daily_refresh_animals, apply_daily_refresh_plants
from .decay import apply_decay
from .end_of_day import drop_all_inventories, reset_units_for_new_day
from .market_queue import process_market_queue
from .player_turn import apply_player_units
from .state import State
from .town import apply_town_consumption


def _tree_where(cond, a, b):
    """条件condに応じて、pytree a/bの対応する葉ごとにjnp.whereを適用する。"""
    return jax.tree.map(lambda x, y: jnp.where(cond, x, y), a, b)


# apply_player_unitsの位置引数の順序に合わせたin_axes。先頭24個(Action/tiles/
# inventory/seeds/shed)はプレイヤー軸(0番目)を持つので0、末尾4個
# (board_size/day/turns_per_day/shed_capacity)は両プレイヤー共通の値なのでNone。
_UNIT_ACTIONS_IN_AXES = (0,) * 24 + (None, None, None, None)


def _run_unit_actions(state, action, board_size, day, turns_per_day, shed_capacity):
    """両プレイヤー分のユニット行動を適用し、Stateの該当フィールドを差し替える。

    apply_player_unitsは1プレイヤー分の処理(プレイヤー間の依存が無い)なので、
    プレイヤー軸(先頭)に対してvmapするだけでよい。
    """
    (
        farmer_pos,
        hands_pos,
        tiles_kind,
        tiles_crop_or_animal,
        tiles_planted_or_placed_day,
        tiles_watered_or_fed_today,
        tiles_consecutive_unwatered_or_unfed,
        tiles_yield_units,
        tiles_max_lifespan_step,
        tiles_fertilized_until_day,
        tiles_cared_today,
        tiles_fertilizer_available,
        tiles_pending_care_bonus,
        farmer_inventory,
        hands_inventory,
        seeds,
        shed,
    ) = jax.vmap(apply_player_units, in_axes=_UNIT_ACTIONS_IN_AXES)(
        action.farmer_op,
        action.farmer_arg_idx,
        action.farmer_n,
        action.hands_op,
        action.hands_arg_idx,
        action.hands_n,
        state.hands_active,
        state.farmer_pos,
        state.hands_pos,
        state.tiles_kind,
        state.tiles_crop_or_animal,
        state.tiles_planted_or_placed_day,
        state.tiles_watered_or_fed_today,
        state.tiles_consecutive_unwatered_or_unfed,
        state.tiles_yield_units,
        state.tiles_max_lifespan_step,
        state.tiles_fertilized_until_day,
        state.tiles_cared_today,
        state.tiles_fertilizer_available,
        state.tiles_pending_care_bonus,
        state.farmer_inventory,
        state.hands_inventory,
        state.seeds,
        state.shed,
        board_size,
        day,
        turns_per_day,
        shed_capacity,
    )

    return state._replace(
        farmer_pos=farmer_pos,
        hands_pos=hands_pos,
        tiles_kind=tiles_kind,
        tiles_crop_or_animal=tiles_crop_or_animal,
        tiles_planted_or_placed_day=tiles_planted_or_placed_day,
        tiles_watered_or_fed_today=tiles_watered_or_fed_today,
        tiles_consecutive_unwatered_or_unfed=tiles_consecutive_unwatered_or_unfed,
        tiles_yield_units=tiles_yield_units,
        tiles_max_lifespan_step=tiles_max_lifespan_step,
        tiles_fertilized_until_day=tiles_fertilized_until_day,
        tiles_cared_today=tiles_cared_today,
        tiles_fertilizer_available=tiles_fertilizer_available,
        tiles_pending_care_bonus=tiles_pending_care_bonus,
        farmer_inventory=farmer_inventory,
        hands_inventory=hands_inventory,
        seeds=seeds,
        shed=shed,
    )


def _run_market(state, action, board_size, hire_mult, shed_capacity):
    """両プレイヤーの市場注文キューを処理し、Stateの該当フィールドを差し替える。"""
    (
        money_a,
        money_b,
        hires_today_a,
        hires_today_b,
        hands_active_a,
        hands_active_b,
        hands_pos_a,
        hands_pos_b,
        unlocked_quadrants_a,
        unlocked_quadrants_b,
        tile_kind_a,
        tile_kind_b,
        seeds_a,
        seeds_b,
        shed_a,
        shed_b,
        market_inventory,
    ) = process_market_queue(
        queue_op_a=action.market_op[0],
        queue_item_a=action.market_arg_idx[0],
        queue_n_a=action.market_n[0],
        queue_op_b=action.market_op[1],
        queue_item_b=action.market_arg_idx[1],
        queue_n_b=action.market_n[1],
        money_a=state.money[0],
        money_b=state.money[1],
        hires_today_a=state.hires_today[0],
        hires_today_b=state.hires_today[1],
        hands_active_a=state.hands_active[0],
        hands_active_b=state.hands_active[1],
        hands_pos_a=state.hands_pos[0],
        hands_pos_b=state.hands_pos[1],
        farmer_pos_a=state.farmer_pos[0],
        farmer_pos_b=state.farmer_pos[1],
        unlocked_quadrants_a=state.unlocked_quadrants[0],
        unlocked_quadrants_b=state.unlocked_quadrants[1],
        tile_kind_a=state.tiles_kind[0],
        tile_kind_b=state.tiles_kind[1],
        seeds_a=state.seeds[0],
        seeds_b=state.seeds[1],
        shed_a=state.shed[0],
        shed_b=state.shed[1],
        market_inventory=state.market_inventory,
        board_size=board_size,
        hire_mult=hire_mult,
        shed_capacity=shed_capacity,
    )

    return state._replace(
        money=jnp.stack([money_a, money_b]),
        hires_today=jnp.stack([hires_today_a, hires_today_b]),
        hands_active=jnp.stack([hands_active_a, hands_active_b]),
        hands_pos=jnp.stack([hands_pos_a, hands_pos_b]),
        unlocked_quadrants=jnp.stack([unlocked_quadrants_a, unlocked_quadrants_b]),
        tiles_kind=jnp.stack([tile_kind_a, tile_kind_b]),
        seeds=jnp.stack([seeds_a, seeds_b]),
        shed=jnp.stack([shed_a, shed_b]),
        market_inventory=market_inventory,
    )


def _refresh_player_day_end(
    tile_kind,
    tile_crop_or_animal,
    tile_planted_or_placed_day,
    tile_watered_or_fed_today,
    tile_consecutive_unwatered_or_unfed,
    tile_yield_units,
    tile_max_lifespan_step,
    tile_fertilized_until_day,
    tile_cared_today,
    tile_fertilizer_available,
    tile_pending_care_bonus,
    farmer_inventory,
    hands_inventory,
    shed,
    weed_key,
    board_size,
    day,
    turns_per_day,
    weed_chance,
    shed_capacity,
):
    """1プレイヤー分の日末処理(日次更新・雑草発生・持ち物払い出し・位置リセット)。"""
    tk2, tc2, twf2, tcu2, tyu2, tmls2 = apply_daily_refresh_plants(
        tile_kind,
        tile_crop_or_animal,
        tile_planted_or_placed_day,
        tile_watered_or_fed_today,
        tile_consecutive_unwatered_or_unfed,
        tile_yield_units,
        tile_max_lifespan_step,
        tile_fertilized_until_day,
        current_day=day,
        turns_per_day=turns_per_day,
    )
    tc3, twf3, tcu3, tyu3, tcared3, tfa3, tpcb3 = apply_daily_refresh_animals(
        tk2,
        tc2,
        tile_planted_or_placed_day,
        twf2,
        tcu2,
        tyu2,
        tile_cared_today,
        tile_fertilizer_available,
        tile_pending_care_bonus,
        day=day,
    )
    tk4 = random_events.spawn_weeds(weed_key, tk2, weed_chance)

    new_shed = drop_all_inventories(farmer_inventory, hands_inventory, shed, shed_capacity)
    (
        farmer_pos,
        hands_pos,
        hands_active,
        hires_today,
        new_farmer_inventory,
        new_hands_inventory,
    ) = reset_units_for_new_day(board_size)

    return (
        tk4,
        tc3,
        tile_planted_or_placed_day,
        twf3,
        tcu3,
        tyu3,
        tmls2,
        tile_fertilized_until_day,
        tcared3,
        tfa3,
        tpcb3,
        farmer_pos,
        hands_pos,
        hands_active,
        hires_today,
        new_farmer_inventory,
        new_hands_inventory,
        new_shed,
    )


# _refresh_player_day_endの位置引数の順序に合わせたin_axes。先頭15個(tiles/
# inventory/shed/weed_key)はプレイヤー軸(0番目)を持つので0、末尾5個
# (board_size/day/turns_per_day/weed_chance/shed_capacity)は両プレイヤー共通の
# 値なのでNone。
_DAY_END_IN_AXES = (0,) * 15 + (None, None, None, None, None)


def _run_end_of_day(
    state,
    board_size,
    day,
    turns_per_day,
    shed_capacity,
    weed_chance,
    shop_unlock_interval,
    max_shop_instances,
):
    """日末処理を計算し、Stateの該当フィールドを差し替える(常に計算し、呼び出し側でis_end_of_dayによって採否を選ぶ)。

    _refresh_player_day_endは1プレイヤー分の処理(プレイヤー間の依存が無い)
    なので、プレイヤー軸(先頭)に対してvmapするだけでよい。
    """
    weed_keys = random_events.daily_rng_keys(state.rng_key, day)  # [3, 2]: p0雑草, p1雑草, 店抽選

    (
        tiles_kind,
        tiles_crop_or_animal,
        tiles_planted_or_placed_day,
        tiles_watered_or_fed_today,
        tiles_consecutive_unwatered_or_unfed,
        tiles_yield_units,
        tiles_max_lifespan_step,
        tiles_fertilized_until_day,
        tiles_cared_today,
        tiles_fertilizer_available,
        tiles_pending_care_bonus,
        farmer_pos,
        hands_pos,
        hands_active,
        hires_today,
        farmer_inventory,
        hands_inventory,
        shed,
    ) = jax.vmap(_refresh_player_day_end, in_axes=_DAY_END_IN_AXES)(
        state.tiles_kind,
        state.tiles_crop_or_animal,
        state.tiles_planted_or_placed_day,
        state.tiles_watered_or_fed_today,
        state.tiles_consecutive_unwatered_or_unfed,
        state.tiles_yield_units,
        state.tiles_max_lifespan_step,
        state.tiles_fertilized_until_day,
        state.tiles_cared_today,
        state.tiles_fertilizer_available,
        state.tiles_pending_care_bonus,
        state.farmer_inventory,
        state.hands_inventory,
        state.shed,
        weed_keys[:2],
        board_size,
        day,
        turns_per_day,
        weed_chance,
        shed_capacity,
    )

    next_day = day + 1
    town_shop_counts = random_events.apply_shop_unlock(
        weed_keys[2], state.town_shop_counts, next_day, shop_unlock_interval, max_shop_instances
    )

    return state._replace(
        tiles_kind=tiles_kind,
        tiles_crop_or_animal=tiles_crop_or_animal,
        tiles_planted_or_placed_day=tiles_planted_or_placed_day,
        tiles_watered_or_fed_today=tiles_watered_or_fed_today,
        tiles_consecutive_unwatered_or_unfed=tiles_consecutive_unwatered_or_unfed,
        tiles_yield_units=tiles_yield_units,
        tiles_max_lifespan_step=tiles_max_lifespan_step,
        tiles_fertilized_until_day=tiles_fertilized_until_day,
        tiles_cared_today=tiles_cared_today,
        tiles_fertilizer_available=tiles_fertilizer_available,
        tiles_pending_care_bonus=tiles_pending_care_bonus,
        farmer_pos=farmer_pos,
        hands_pos=hands_pos,
        hands_active=hands_active,
        hires_today=hires_today,
        farmer_inventory=farmer_inventory,
        hands_inventory=hands_inventory,
        shed=shed,
        town_shop_counts=town_shop_counts,
    )


def step(
    state: State,
    action: Action,
    board_size=10,
    turns_per_day=24,
    shed_capacity=100,
    weed_chance=0.005,
    shop_unlock_interval=3,
    shop_sell_interval=4,
    center_sell_interval=24,
    hire_mult=1,
    max_shop_instances=8,
    episode_steps=720,
    compute_end_of_day=True,
):
    """1ターン進める。

    Args:
        state: 現在のState。
        action: このターンの両プレイヤーの行動。
        board_size, turns_per_day, ...: kaggriculture.jsonのconfigurationに対応する
            設定値。jax.jitで包む場合、board_sizeは配列のshapeを決めるため
            static_argnamesに含める必要がある。
        compute_end_of_day: 静的なbool。日末処理を計算するかどうか
            (step_batch_lockstepが、日末でないと分かっているターンでFalseを
            渡して省略するために使う)。日末ターンでFalseにすると結果が壊れる
            ため、通常はTrue(デフォルト)のまま使うこと。

    Returns:
        (new_state, reward, done)。rewardは[2]で終端ターンのみ両者のmoney、
        それ以外は0。doneはこのターンで終了したかどうか。
    """
    day = state.step // turns_per_day

    state = _run_unit_actions(state, action, board_size, day, turns_per_day, shed_capacity)
    state = _run_market(state, action, board_size, hire_mult, shed_capacity)

    new_market_inventory = apply_town_consumption(
        state.market_inventory,
        state.town_shop_counts,
        state.step,
        shop_sell_interval,
        center_sell_interval,
    )
    state = state._replace(market_inventory=new_market_inventory)

    new_tiles_kind, new_tiles_crop_or_animal, new_tiles_yield_units = apply_decay(
        state.tiles_kind,
        state.tiles_crop_or_animal,
        state.tiles_max_lifespan_step,
        state.tiles_yield_units,
        state.step,
    )
    state = state._replace(
        tiles_kind=new_tiles_kind,
        tiles_crop_or_animal=new_tiles_crop_or_animal,
        tiles_yield_units=new_tiles_yield_units,
    )

    if compute_end_of_day:
        is_end_of_day = ((state.step + 1) % turns_per_day) == 0
        end_of_day_state = _run_end_of_day(
            state,
            board_size,
            day,
            turns_per_day,
            shed_capacity,
            weed_chance,
            shop_unlock_interval,
            max_shop_instances,
        )
        state = _tree_where(is_end_of_day, end_of_day_state, state)

    next_step = state.step + 1
    state = state._replace(step=next_step)

    done = state.step >= (episode_steps - 1)
    reward = jnp.where(done, state.money, jnp.zeros_like(state.money))

    return state, reward, done


def step_batch(
    state: State,
    action: Action,
    board_size=10,
    turns_per_day=24,
    shed_capacity=100,
    weed_chance=0.005,
    shop_unlock_interval=3,
    shop_sell_interval=4,
    center_sell_interval=24,
    hire_mult=1,
    max_shop_instances=8,
    episode_steps=720,
):
    """step()のバッチ版。step()自体はバッチ軸を持たない1局分のStateしか
    受け取れない(state.pyのモジュールdocstring参照)ので、B局分を
    まとめて進めたい時はこちらを使う。

    Args:
        state: [B, 2, ...] 形のState(reset()の戻り値と同じ形)。
        action: [B, 2, ...] 形のAction。
        board_size, turns_per_day, ...: step()と同じ(全局共通の静的値なので
            vmapの対象に含めない)。

    Returns:
        step()と同じ形の (new_state, reward, done) に、先頭のバッチ軸Bが付いたもの。
    """
    return jax.vmap(
        step, in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None)
    )(
        state,
        action,
        board_size,
        turns_per_day,
        shed_capacity,
        weed_chance,
        shop_unlock_interval,
        shop_sell_interval,
        center_sell_interval,
        hire_mult,
        max_shop_instances,
        episode_steps,
    )


def step_batch_lockstep(
    state: State,
    action: Action,
    board_size=10,
    turns_per_day=24,
    shed_capacity=100,
    weed_chance=0.005,
    shop_unlock_interval=3,
    shop_sell_interval=4,
    center_sell_interval=24,
    hire_mult=1,
    max_shop_instances=8,
    episode_steps=720,
):
    """step_batch()の最適化版。バッチ内の全局が同じstepを共有している
    (reset()でB局同時に開始し、以後これだけで進める通常の学習ループ)という
    前提を置き、日末処理をバッチ全体で1回のjax.lax.condにより日末ターンの
    時だけ実行する(局ごとにwhereで選ぶstep_batch()より、日末以外のターンが
    軽くなる)。

    jax.lax.condは両方の分岐をトレース・コンパイルするため、初回コンパイルは
    step_batchより重くなる(長時間の学習ループでは回収できる)。

    局ごとにstepがずれている場合(個別リセット等)は使えない
    (判定にstate.step[0]だけを見るため。通常はstep_batch()を使うこと)。

    Args:
        state, action, board_size, ...: step_batch()と同じ。

    Returns:
        step_batch()と同じ形の (new_state, reward, done)。
    """
    is_end_of_day = ((state.step[0] + 1) % turns_per_day) == 0

    def run(compute_end_of_day):
        fn = functools.partial(
            step,
            board_size=board_size,
            turns_per_day=turns_per_day,
            shed_capacity=shed_capacity,
            weed_chance=weed_chance,
            shop_unlock_interval=shop_unlock_interval,
            shop_sell_interval=shop_sell_interval,
            center_sell_interval=center_sell_interval,
            hire_mult=hire_mult,
            max_shop_instances=max_shop_instances,
            episode_steps=episode_steps,
            compute_end_of_day=compute_end_of_day,
        )
        return jax.vmap(fn)(state, action)

    return jax.lax.cond(is_end_of_day, lambda: run(True), lambda: run(False))
