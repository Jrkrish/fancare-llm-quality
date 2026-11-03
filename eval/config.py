"""Loads eval/config.toml, applies a profile and overrides."""
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from app.config import ROOT

CONFIG_PATH = ROOT / "eval" / "config.toml"
REPORTS_DIR = ROOT / "reports"


@dataclass
class EvalConfig:
    profile: str
    dataset: Path
    runs_per_case: int
    min_case_pass_rate: float
    unstable_std: float
    cases_per_category: int
    max_error_rate: float
    judge_provider: str
    judge_model: str
    judge_min_interval_s: float
    judge_metrics: list[str]
    thresholds: dict[str, float]
    hallucination_faithfulness_below: float
    gates: dict[str, float]
    max_hallucination_rate: float


def load_eval_config(profile: str = "full", path: Path = CONFIG_PATH, **overrides) -> EvalConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if profile not in raw.get("profiles", {}):
        raise ValueError(f"Unknown eval profile '{profile}'. Available: {sorted(raw['profiles'])}")
    prof = raw["profiles"][profile]
    run = {**raw["run"], **{k: v for k, v in prof.items() if k in raw["run"]}}
    gates = dict(raw["gates"])

    cfg = EvalConfig(
        profile=profile,
        dataset=ROOT / run["dataset"],
        runs_per_case=int(os.getenv("EVAL_RUNS") or run["runs_per_case"]),
        min_case_pass_rate=run["min_case_pass_rate"],
        unstable_std=run["unstable_std"],
        cases_per_category=run["cases_per_category"],
        max_error_rate=run["max_error_rate"],
        judge_provider=os.getenv("JUDGE_PROVIDER") or raw["judge"]["provider"],
        judge_model=os.getenv("JUDGE_MODEL") or raw["judge"]["model"],
        judge_min_interval_s=raw["judge"]["min_interval_s"],
        judge_metrics=list(prof.get("judge_metrics", raw["metrics"]["judge_metrics"])),
        thresholds=dict(raw["thresholds"]),
        hallucination_faithfulness_below=raw["hallucination"]["faithfulness_below"],
        max_hallucination_rate=gates.pop("max_hallucination_rate"),
        gates=gates,
    )
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg
