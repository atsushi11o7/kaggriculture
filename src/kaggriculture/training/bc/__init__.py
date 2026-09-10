"""Behavior Cloning用の目的関数と、将来のデータ処理・学習ループ。"""

from kaggriculture.training.bc.objective import mean_token_nll

__all__ = ["mean_token_nll"]
