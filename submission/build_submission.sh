#!/usr/bin/env bash
# submission/(gitignore対象)配下に提出用tar.gzを組み立てる。
#
# 使い方:
#   ./submission/build_submission.sh <opening用checkpoint.pt> <day12+用checkpoint.pt> <出力名(拡張子無し)>
#
# 例:
#   ./submission/build_submission.sh \
#     submission/exports/opening_day11_decem_v1_best.pt \
#     submission/exports/day12plus_decem_v1_best.pt \
#     02_two_stage_bc_v1
#
# main.pyがtar.gzのルートに来るよう、submission/build/直下に組み立ててから固める
# (kaggle-environmentsはmain.pyと同じディレクトリをsys.pathに追加するため、
# kaggriculture/パッケージもmain.pyと同じ階層に置く)。

set -euo pipefail

OPENING_CHECKPOINT="${1:?opening checkpoint path required}"
DAY12PLUS_CHECKPOINT="${2:?day12+ checkpoint path required}"
NAME="${3:?output name (no extension) required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
DIST_DIR="$SCRIPT_DIR/dist"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/kaggriculture/policy/common" "$BUILD_DIR/kaggriculture/policy/torch" \
  "$BUILD_DIR/kaggriculture/policy/endgame" "$BUILD_DIR/kaggriculture/policy/opening" \
  "$BUILD_DIR/kaggriculture/rules" "$DIST_DIR"

# --- policy/推論経路だけが必要とするソースを最小限vendorする ---
# GPUシミュレータ本体とJAX backendは含めない。openingはday<=11をBCモデルで直接
# 推論するようになったため、旧テープ方式(pre_shop/pre_shop_tape/post_shop_tape/
# bridge_tape/tape_bank/planner/targets/demand)はもう不要。土地購入の反応的な
# 安全網としてland_rule.pyだけ残す。
touch "$BUILD_DIR/kaggriculture/__init__.py"
touch "$BUILD_DIR/kaggriculture/policy/__init__.py"
touch "$BUILD_DIR/kaggriculture/policy/common/__init__.py"
touch "$BUILD_DIR/kaggriculture/policy/torch/__init__.py"
touch "$BUILD_DIR/kaggriculture/policy/endgame/__init__.py"
touch "$BUILD_DIR/kaggriculture/policy/opening/__init__.py"
touch "$BUILD_DIR/kaggriculture/rules/__init__.py"

cp "$REPO_ROOT/src/kaggriculture/policy/common/"*.py "$BUILD_DIR/kaggriculture/policy/common/"
cp "$REPO_ROOT/src/kaggriculture/policy/torch/"*.py "$BUILD_DIR/kaggriculture/policy/torch/"
cp "$REPO_ROOT/src/kaggriculture/policy/endgame/common.py" "$BUILD_DIR/kaggriculture/policy/endgame/"
cp "$REPO_ROOT/src/kaggriculture/policy/endgame/torch.py" "$BUILD_DIR/kaggriculture/policy/endgame/"
cp "$REPO_ROOT/src/kaggriculture/policy/opening/land_rule.py" "$BUILD_DIR/kaggriculture/policy/opening/"
cp "$REPO_ROOT/src/kaggriculture/rules/constants.py" "$BUILD_DIR/kaggriculture/rules/"
cp "$REPO_ROOT/src/kaggriculture/rules/game_params.py" "$BUILD_DIR/kaggriculture/rules/"

cp "$OPENING_CHECKPOINT" "$BUILD_DIR/model_weights_opening.pt"
cp "$DAY12PLUS_CHECKPOINT" "$BUILD_DIR/model_weights_day12plus.pt"
cp "$SCRIPT_DIR/main.py" "$BUILD_DIR/main.py"

OUT="$DIST_DIR/${NAME}.tar.gz"
tar -czf "$OUT" -C "$BUILD_DIR" main.py model_weights_opening.pt model_weights_day12plus.pt kaggriculture

echo "built: $OUT"
tar -tzf "$OUT"
