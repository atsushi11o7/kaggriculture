"""スロット種別ごとのLightGBM候補ranker。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from kaggriculture.training.gbdt.dataset import RankingData
from kaggriculture.training.gbdt.features import FEATURE_NAMES, FEATURE_VERSION


@dataclass(frozen=True)
class RankerConfig:
    """各LightGBM rankerへ共通で渡す設定。"""

    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 40
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    random_state: int = 0
    n_jobs: int = -1
    early_stopping_rounds: int = 50


class GBDTRanker:
    """unit、市場、数量の候補を別々に順位付けする教師モデル。"""

    def __init__(self, models: dict[str, lgb.Booster], config: RankerConfig):
        self.models = models
        self.config = config

    @classmethod
    def fit(
        cls,
        datasets: dict[str, RankingData],
        config: RankerConfig,
        validation: dict[str, RankingData] | None = None,
    ) -> GBDTRanker:
        """正解候補をrelevance 1とするLambdaRankを学習する。"""
        models = {}
        parameters = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "label_gain": [0, 1],
            "learning_rate": config.learning_rate,
            "num_leaves": config.num_leaves,
            "min_data_in_leaf": config.min_child_samples,
            "bagging_fraction": config.subsample,
            "bagging_freq": 1,
            "feature_fraction": config.colsample_bytree,
            "lambda_l2": config.reg_lambda,
            "seed": config.random_state,
            "num_threads": config.n_jobs,
            "verbosity": -1,
        }
        for kind, data in datasets.items():
            dataset = lgb.Dataset(
                data.features,
                label=data.labels,
                group=data.groups,
                feature_name=list(FEATURE_NAMES),
                free_raw_data=False,
            )
            valid = validation.get(kind) if validation is not None else None
            valid_sets = None
            callbacks = None
            if valid is not None:
                valid_sets = [
                    lgb.Dataset(
                        valid.features,
                        label=valid.labels,
                        group=valid.groups,
                        reference=dataset,
                        feature_name=list(FEATURE_NAMES),
                    )
                ]
                callbacks = [lgb.early_stopping(config.early_stopping_rounds, verbose=False)]
            models[kind] = lgb.train(
                parameters,
                dataset,
                num_boost_round=config.n_estimators,
                valid_sets=valid_sets,
                callbacks=callbacks,
            )
        return cls(models, config)

    def top1_accuracy(self, kind: str, data: RankingData) -> float:
        """queryごとの最上位候補が教師選択と一致する割合を返す。"""
        scores = self.score(kind, data.features)
        correct = 0
        offset = 0
        for size in data.groups:
            end = offset + int(size)
            correct += int(np.argmax(scores[offset:end]) == np.argmax(data.labels[offset:end]))
            offset = end
        return correct / len(data.groups)

    def score(self, kind: str, features: np.ndarray) -> np.ndarray:
        """同一query内の各合法候補へ順位スコアを返す。"""
        model = self.models.get(kind)
        if model is None:
            return np.zeros(len(features), dtype=np.float32)
        return np.asarray(model.predict(features, num_threads=1), dtype=np.float32)

    def save(self, directory: Path) -> None:
        """ranker群と特徴量契約を保存する。"""
        directory.mkdir(parents=True, exist_ok=True)
        for kind, model in self.models.items():
            model.save_model(directory / f"{kind}.txt")
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "trainer": "gbdt_ranker",
                    "feature_version": FEATURE_VERSION,
                    "feature_names": FEATURE_NAMES,
                    "config": asdict(self.config),
                    "kinds": sorted(self.models),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> GBDTRanker:
        """保存したranker群を特徴量契約の検証後に復元する。"""
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if (
            metadata["feature_version"] != FEATURE_VERSION
            or tuple(metadata["feature_names"]) != FEATURE_NAMES
        ):
            raise ValueError("GBDT feature contract does not match this code")
        config = RankerConfig(**metadata["config"])
        models = {}
        for kind in metadata["kinds"]:
            models[kind] = lgb.Booster(model_file=directory / f"{kind}.txt")
        return cls(models, config)
