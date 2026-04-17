#!/usr/bin/env python3
"""Standalone RPO analysis for NFL advanced passing data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import nflreadpy as nfl
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: nflreadpy. Install with `pip install nflreadpy pandas`."
    ) from exc


REQUIRED_COLUMNS = [
    "player",
    "team",
    "season",
    "rpo_plays",
    "rpo_yards",
    "rpo_pass_att",
    "rpo_pass_yards",
    "rpo_rush_att",
    "rpo_rush_yards",
]


def safe_div(numerator: float, denominator: float) -> float:
    """Return numerator/denominator or NaN when denominator is zero."""
    return numerator / denominator if denominator else np.nan


def load_rpo_data(seasons: Iterable[int]) -> pd.DataFrame:
    """Load season-level passing advanced stats that include RPO fields."""
    if not hasattr(nfl, "load_pfr_advstats"):
        raise RuntimeError(
            "This nflreadpy version does not expose load_pfr_advstats; "
            "upgrade nflreadpy and retry."
        )

    data = nfl.load_pfr_advstats(
        stat_type="pass",
        summary_level="season",
        seasons=list(seasons),
    )

    if hasattr(data, "to_dicts"):
        df = pd.DataFrame(data.to_dicts())
    else:
        df = pd.DataFrame(data)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected RPO columns: {missing}")

    numeric_cols = [c for c in REQUIRED_COLUMNS if c not in {"player", "team"}]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    return df[REQUIRED_COLUMNS].copy()


def build_league_summary(df: pd.DataFrame) -> pd.Series:
    """Compute league-level RPO summary metrics."""
    total_rpo_plays = df["rpo_plays"].sum()
    total_rpo_pass_att = df["rpo_pass_att"].sum()
    total_rpo_rush_att = df["rpo_rush_att"].sum()
    total_rpo_yards = df["rpo_yards"].sum()

    return pd.Series(
        {
            "total_rpo_plays": int(total_rpo_plays),
            "total_rpo_yards": int(total_rpo_yards),
            "rpo_pass_share_pct": round(100 * safe_div(total_rpo_pass_att, total_rpo_plays), 2),
            "rpo_rush_share_pct": round(100 * safe_div(total_rpo_rush_att, total_rpo_plays), 2),
            "yards_per_rpo_play": round(safe_div(total_rpo_yards, total_rpo_plays), 2),
            "yards_per_rpo_pass": round(
                safe_div(df["rpo_pass_yards"].sum(), total_rpo_pass_att), 2
            ),
            "yards_per_rpo_rush": round(
                safe_div(df["rpo_rush_yards"].sum(), total_rpo_rush_att), 2
            ),
        }
    )


def build_season_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute season-level RPO summaries."""
    summary = (
        df.groupby("season", as_index=False)[
            ["rpo_plays", "rpo_yards", "rpo_pass_att", "rpo_rush_att"]
        ]
        .sum()
        .sort_values("season")
    )
    summary["yards_per_play"] = summary["rpo_yards"] / summary["rpo_plays"].replace(0, np.nan)
    summary["pass_share_pct"] = 100 * summary["rpo_pass_att"] / summary["rpo_plays"].replace(0, np.nan)
    summary["rush_share_pct"] = 100 * summary["rpo_rush_att"] / summary["rpo_plays"].replace(0, np.nan)
    return summary.round(2)


def build_top_players(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Return top players by RPO play volume with efficiency fields."""
    cols = [
        "season",
        "player",
        "team",
        "rpo_plays",
        "rpo_yards",
        "rpo_pass_att",
        "rpo_rush_att",
    ]
    top = df.sort_values(["rpo_plays", "rpo_yards"], ascending=False)[cols].head(top_n).copy()
    top["yards_per_play"] = top["rpo_yards"] / top["rpo_plays"].replace(0, np.nan)
    top["pass_share_pct"] = 100 * top["rpo_pass_att"] / top["rpo_plays"].replace(0, np.nan)
    return top.round(2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze NFL RPO trends.")
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--end-season", type=int, default=2025)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save CSV summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_season > args.end_season:
        raise SystemExit("start-season cannot be greater than end-season")

    seasons = range(args.start_season, args.end_season + 1)
    df = load_rpo_data(seasons)

    league_summary = build_league_summary(df)
    season_summary = build_season_summary(df)
    top_players = build_top_players(df, args.top_n)

    print("\n=== LEAGUE RPO SUMMARY ===")
    print(league_summary.to_string())
    print("\n=== RPO BY SEASON ===")
    print(season_summary.to_string(index=False))
    print(f"\n=== TOP {args.top_n} PLAYERS BY RPO PLAYS ===")
    print(top_players.to_string(index=False))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        league_summary.to_frame("value").to_csv(args.output_dir / "league_rpo_summary.csv")
        season_summary.to_csv(args.output_dir / "season_rpo_summary.csv", index=False)
        top_players.to_csv(args.output_dir / "top_rpo_players.csv", index=False)
        print(f"\nSaved CSV outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
