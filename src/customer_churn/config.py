from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        p = Path(path) if path else project_root() / "config" / "config.yaml"
        cfg = cls(raw=yaml.safe_load(open(p)))
        cfg.validate()
        return cfg

    def validate(self) -> None:
        c = self.raw["costs"]
        if c["missed_churner"] <= c["intervention"]:
            raise ValueError(
                "missed_churner must exceed intervention cost, otherwise the "
                "cost-optimal action is to contact nobody and the threshold is degenerate"
            )
        if not self.raw["banned_features"]:
            raise ValueError("banned_features must not be empty")

        overlap = set(self.banned_features) & set(self.feature_names)
        if overlap:
            raise ValueError(f"banned features listed as model features: {sorted(overlap)}")

        dupes = set(self.numeric_features) & set(self.categorical_features)
        if dupes:
            raise ValueError(f"column declared both numeric and categorical: {sorted(dupes)}")

        # YAML 1.1 turns an unquoted `No` into the boolean False. If that happens the
        # fill value stops matching the category vocabulary and the model silently sees
        # a new level at serve time -- the same class of bug this schema exists to stop.
        unquoted = {k: v for k, v in self.null_fill.items() if not isinstance(v, str)}
        if unquoted:
            raise ValueError(
                f"null_fill values must be quoted strings in config.yaml; "
                f"these parsed as non-strings: {unquoted}"
            )

    @property
    def target(self) -> str:
        return self.raw["target"]["column"]

    @property
    def positive_class(self) -> str:
        return self.raw["target"]["positive_class"]

    @property
    def banned_features(self) -> list[str]:
        return list(self.raw["banned_features"])

    # -- schema ---------------------------------------------------------------
    @property
    def numeric_features(self) -> list[str]:
        return list(self.raw["schema"]["numeric_features"])

    @property
    def categorical_features(self) -> list[str]:
        return list(self.raw["schema"]["categorical_features"])

    @property
    def feature_names(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    @property
    def null_fill(self) -> dict[str, str]:
        return dict(self.raw["schema"]["null_fill"])

    @property
    def boolean_labels(self) -> tuple[str, str]:
        s = self.raw["schema"]
        return s["boolean_true"], s["boolean_false"]

    def path(self, key: str) -> Path:
        return project_root() / self.raw["paths"][key]

    def data_path(self, key: str) -> Path:
        return project_root() / self.raw["data"][key]


def library_versions() -> dict[str, str]:
    """Versions that produced the artefact.

    A scikit-learn pickle is not portable across versions. Recording this makes a
    mismatch a legible message rather than an AttributeError from deep inside
    joblib -- which is how it presents when CI or a hosted app installs a newer
    scikit-learn than the one that did the training.
    """
    import platform

    versions = {"python": platform.python_version()}
    for name in ("sklearn", "numpy", "scipy", "pandas", "joblib", "xgboost"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            versions[name] = "absent"
    return versions
