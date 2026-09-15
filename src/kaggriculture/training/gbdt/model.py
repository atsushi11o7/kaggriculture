"""スロット種別ごとのLightGBM候補ranker。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from kaggriculture.training.gbdt.dataset import RankingData, RankingFiles, open_ranking_data
from kaggriculture.training.gbdt.features import FEATURE_NAMES, FEATURE_VERSION


class _MappedSequence(lgb.Sequence):
    """LightGBMのDataset構築時にmemmapから必要な行だけ読む。"""

    batch_size = 8192

    def __init__(self, features: np.ndarray):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        values = np.asarray(self.features[index])
        # LightGBMのcolumn samplingは単一行をfloat64で要求する。
        return values.astype(np.float64) if isinstance(index, (int, np.integer)) else values


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
        datasets: dict[str, RankingData] | RankingFiles,
        config: RankerConfig,
        validation: dict[str, RankingData] | RankingFiles | None = None,
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
        kinds = (
            [kind for kind, info in datasets.kinds.items() if info["rows"]]
            if isinstance(datasets, RankingFiles)
            else list(datasets)
        )
        for kind in kinds:
            data = (
                open_ranking_data(datasets, kind)
                if isinstance(datasets, RankingFiles)
                else datasets[kind]
            )
            dataset = lgb.Dataset(
                _MappedSequence(data.features)
                if isinstance(datasets, RankingFiles)
                else data.features,
                label=data.labels,
                group=data.groups,
                weight=data.weights,
                feature_name=list(FEATURE_NAMES),
                free_raw_data=True,
            )
            valid = (
                open_ranking_data(validation, kind)
                if isinstance(validation, RankingFiles) and validation.kinds[kind]["rows"]
                else validation.get(kind)
                if isinstance(validation, dict)
                else None
            )
            valid_sets = None
            callbacks = None
            if valid is not None:
                valid_sets = [
                    lgb.Dataset(
                        _MappedSequence(valid.features)
                        if isinstance(validation, RankingFiles)
                        else valid.features,
                        label=valid.labels,
                        group=valid.groups,
                        weight=valid.weights,
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
        """query境界で分割し、予測スコア全件をRAMへ置かずに評価する。"""
        correct = 0
        offset = 0
        groups = data.groups
        group_index = 0
        while group_index < len(groups):
            start = offset
            end_group = group_index
            while end_group < len(groups) and offset - start < 100_000:
                offset += int(groups[end_group])
                end_group += 1
            scores = self.score(kind, data.features[start:offset])
            relative = 0
            for size in groups[group_index:end_group]:
                end = relative + int(size)
                correct += int(
                    np.argmax(scores[relative:end])
                    == np.argmax(data.labels[start + relative : start + end])
                )
                relative = end
            group_index = end_group
        return correct / len(groups)

    def class_top1_report(self, kind: str, data: RankingData) -> dict:
        """market_op向け: クラス別top1精度・macro精度・予測分布を返す。

        全体top1(top1_accuracy)はSTOP/HIREのような多数派クラスの高精度に
        隠れて、BUY_SEED等の少数派だが重要なクラスの性能が見えない。この
        メソッドはRankingData(features/labels)だけから、期待した候補
        (正解ラベル)とモデルの予測それぞれのop名をone-hot特徴から復元し、
        クラスごとの精度・出現数・モデルが実際に選んだクラスの分布を返す。

        精度は2種類計算する。`per_class_accuracy`(exact)は候補インデックスの
        完全一致(top1_accuracyと同じ基準。作物まで一致が必要)、
        `per_class_op_accuracy`(op)は復元したop名同士の一致だけを見る
        (BUY_SEEDが正解でBUY_SEED(別の作物)を選んでも正解扱い)。opレベルは
        「その種類の行動を選ぶという判断自体ができたか」を、exactは「候補の
        細部まで正しいか」を見る。両方の値を混同しないよう分けて保持する。
        """
        from kaggriculture.training.gbdt.features import classify_market_op_row

        exact_correct_by_class: dict[str, int] = {}
        op_correct_by_class: dict[str, int] = {}
        total_by_class: dict[str, int] = {}
        predicted_by_class: dict[str, int] = {}
        offset = 0
        groups = data.groups
        group_index = 0
        while group_index < len(groups):
            start = offset
            end_group = group_index
            while end_group < len(groups) and offset - start < 100_000:
                offset += int(groups[end_group])
                end_group += 1
            scores = self.score(kind, data.features[start:offset])
            relative = 0
            for size in groups[group_index:end_group]:
                size = int(size)
                end = relative + size
                group_scores = scores[relative:end]
                group_labels = np.asarray(data.labels[start + relative : start + end])
                group_rows = np.asarray(data.features[start + relative : start + end])
                expert_index = int(np.argmax(group_labels))
                predicted_index = int(np.argmax(group_scores))
                expert_class = classify_market_op_row(group_rows[expert_index])
                predicted_class = classify_market_op_row(group_rows[predicted_index])
                total_by_class[expert_class] = total_by_class.get(expert_class, 0) + 1
                if predicted_index == expert_index:
                    exact_correct_by_class[expert_class] = (
                        exact_correct_by_class.get(expert_class, 0) + 1
                    )
                if predicted_class == expert_class:
                    op_correct_by_class[expert_class] = op_correct_by_class.get(expert_class, 0) + 1
                predicted_by_class[predicted_class] = predicted_by_class.get(predicted_class, 0) + 1
                relative = end
            group_index = end_group
        per_class_accuracy = {
            cls: exact_correct_by_class.get(cls, 0) / total for cls, total in total_by_class.items()
        }
        per_class_op_accuracy = {
            cls: op_correct_by_class.get(cls, 0) / total for cls, total in total_by_class.items()
        }
        macro_accuracy = (
            float(np.mean(list(per_class_accuracy.values()))) if per_class_accuracy else 0.0
        )
        macro_op_accuracy = (
            float(np.mean(list(per_class_op_accuracy.values()))) if per_class_op_accuracy else 0.0
        )
        return {
            "per_class_accuracy": per_class_accuracy,
            "per_class_op_accuracy": per_class_op_accuracy,
            "per_class_count": total_by_class,
            "macro_accuracy": macro_accuracy,
            "macro_op_accuracy": macro_op_accuracy,
            "predicted_distribution": predicted_by_class,
        }

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
