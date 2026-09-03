"""ゴールデントレース(CPU版の実行記録)とGPU移植版を突き合わせる。

data/golden_traces/*_deterministic.json (雑草発生・店の抽選を設定で無効化した
トレース)を読み込み、記録されている行動をそのままstep()に流し込んで、
各ターンの状態がCPU版と完全に一致するか比較する。トレースJSON↔State/Actionの
変換はtrace_codec.pyが担う。
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trace_codec import build_action, build_state  # noqa: E402

from kaggriculture.simulator import constants as C  # noqa: E402
from kaggriculture.simulator.state import State  # noqa: E402
from kaggriculture.simulator.step import step  # noqa: E402

# board_sizeは配列のshapeを決めるため、静的引数として扱う。
_jitted_step = jax.jit(step, static_argnames=["board_size"])

TRACE_DIR = Path(__file__).resolve().parent.parent / "data" / "golden_traces"

# タイルの種類ごとに「意味を持つフィールド」だけを比較する。該当しないフィールドは
# 元コードでも二度と読まれない残存値になりうる(例: WEEDになった後のplanted_day)ため、
# 一致していなくても実害が無く、比較対象から除外する。
_PLANT_ONLY_FIELDS = (
    "tiles_crop_or_animal",
    "tiles_planted_or_placed_day",
    "tiles_watered_or_fed_today",
    "tiles_consecutive_unwatered_or_unfed",
    "tiles_yield_units",
    "tiles_max_lifespan_step",
    "tiles_fertilized_until_day",
)
_ANIMAL_ONLY_FIELDS = (
    "tiles_crop_or_animal",
    "tiles_planted_or_placed_day",
    "tiles_watered_or_fed_today",
    "tiles_consecutive_unwatered_or_unfed",
    "tiles_yield_units",
    "tiles_cared_today",
    "tiles_fertilizer_available",
    "tiles_pending_care_bonus",
)
_TILE_FIELDS = set(_PLANT_ONLY_FIELDS) | set(_ANIMAL_ONLY_FIELDS)


def diff_state(expected: State, actual: State):
    """2つのStateを比較し、食い違うフィールド名のリストを返す(タイルのマスク考慮)。"""
    mismatches = []
    for field in State._fields:
        if field == "rng_key":
            continue  # このスクリプトでは未使用
        e = getattr(expected, field)
        a = getattr(actual, field)
        if field not in _TILE_FIELDS:
            if not jnp.array_equal(e, a):
                mismatches.append(field)
            continue

        # タイルフィールド: kind(と動物の有無)に応じて意味のある場所だけ比較する
        is_plant = expected.tiles_kind == C.TILE_PLANT
        is_animal_structure = (expected.tiles_kind == C.TILE_COOP) | (
            expected.tiles_kind == C.TILE_PASTURE
        )
        has_animal = expected.tiles_crop_or_animal >= 0
        relevant = jnp.zeros_like(is_plant)
        if field in _PLANT_ONLY_FIELDS:
            relevant = relevant | is_plant
        if field in _ANIMAL_ONLY_FIELDS:
            relevant = relevant | (is_animal_structure & has_animal)
        if field == "tiles_crop_or_animal":
            # 空のCOOP/PASTUREも「-1であること」自体は常に意味を持つ
            relevant = relevant | is_animal_structure

        if not jnp.array_equal(jnp.where(relevant, e, 0), jnp.where(relevant, a, 0)):
            mismatches.append(field)
    return mismatches


def main(trace_name, max_steps=None):
    """1本のトレースを検証する。戻り値は (検証ターン数, 食い違いターン数)。"""
    with open(TRACE_DIR / f"{trace_name}.json") as f:
        data = json.load(f)
    cfg = data["configuration"]
    board_size = cfg["boardSize"]
    step_kwargs = dict(
        board_size=board_size,
        turns_per_day=cfg["turnsPerDay"],
        shed_capacity=cfg["shedCapacity"],
        weed_chance=cfg["weedSpawnChance"],
        shop_unlock_interval=cfg["townShopUnlockInterval"],
        shop_sell_interval=cfg["townShopSellInterval"],
        center_sell_interval=cfg["townCenterSellInterval"],
        hire_mult=cfg["farmHandCostMult"],
        episode_steps=cfg["episodeSteps"],
    )

    steps = data["steps"]
    if max_steps:
        steps = steps[: max_steps + 1]

    state = build_state(steps[0][0]["observation"], steps[0][1]["observation"], board_size)
    print(f"初期状態を読み込み: step={int(state.step)}")

    n_checked = 0
    n_mismatch = 0
    for i in range(1, len(steps)):
        action0 = steps[i][0]["action"]
        action1 = steps[i][1]["action"]
        action = build_action(action0, action1)

        state, reward, done = _jitted_step(state, action, **step_kwargs)

        expected = build_state(steps[i][0]["observation"], steps[i][1]["observation"], board_size)
        mismatches = diff_state(expected, state)

        expected_reward = jnp.array(
            [steps[i][0]["reward"], steps[i][1]["reward"]], dtype=jnp.float32
        )
        expected_done = steps[i][0]["status"] == "DONE"
        if not jnp.allclose(expected_reward, reward, atol=1e-3):
            mismatches.append(f"reward(expected={expected_reward}, actual={reward})")
        if bool(done) != expected_done:
            mismatches.append(f"done(expected={expected_done}, actual={bool(done)})")

        n_checked += 1
        if mismatches:
            n_mismatch += 1
            print(f"[step {i}] 食い違い: {mismatches}")
            if n_mismatch >= 5:
                print("...(5件で打ち切り)")
                break

    print(f"{n_checked}ターン中 {n_mismatch}ターンで食い違い")
    return n_checked, n_mismatch


def run_all_deterministic_traces():
    """data/golden_traces/*_deterministic.json を全件・全ターン検証する。

    厳密突き合わせできるのは(雑草発生・店の抽選を無効化した)_deterministic系
    トレースのみ(モジュールdocstring参照)。CIや回帰確認用の既定動作として、
    引数無しで実行した場合はこちらを使う(個別に短縮実行したい場合は
    `main(trace_name, max_steps)` を直接呼ぶ)。
    """
    trace_paths = sorted(TRACE_DIR.glob("*_deterministic.json"))
    results = []
    for path in trace_paths:
        print(f"\n=== {path.stem} ===")
        n_checked, n_mismatch = main(path.stem)
        results.append((path.stem, n_checked, n_mismatch))

    print("\n=== まとめ ===")
    ok = True
    for name, n_checked, n_mismatch in results:
        mark = "OK" if n_mismatch == 0 else "NG"
        print(f"[{mark}] {name}: {n_checked}ターン中{n_mismatch}ターンで食い違い")
        ok = ok and n_mismatch == 0
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else None
        _, n_mismatch = main(name, max_steps=max_steps)
        if n_mismatch:
            sys.exit(1)
    else:
        run_all_deterministic_traces()
