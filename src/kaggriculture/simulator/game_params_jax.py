"""game_params.pyの一部テーブルのjnp版。

crop_actions.py/daily_refresh.py(GPUシミュレータ)だけが使う。同じテーブルを
別々のファイルで作り直すと値がズレた時に気づきにくいので、game_params.pyの
plain tupleから作る。policy/推論経路はgame_params.pyだけを使い、このモジュールは
importしない(jax[cuda12]一式を巻き込まないため)。
"""

import jax.numpy as jnp

from kaggriculture.rules.game_params import CROP_FIRST_YIELD_DAY, CROP_IS_ONGOING, CROP_MAX_YIELD

CROP_IS_ONGOING_JAX = jnp.array(CROP_IS_ONGOING)
CROP_FIRST_YIELD_DAY_JAX = jnp.array(CROP_FIRST_YIELD_DAY, dtype=jnp.int32)
CROP_MAX_YIELD_JAX = jnp.array(CROP_MAX_YIELD, dtype=jnp.int32)
