"""DuckDB ETL runner.

The original pipeline lived in SQL Server Management Studio: an import wizard, five
scripts run by hand, then an Excel export read by the notebook. None of that is
reproducible from a clone. This module runs the same logic against an embedded DuckDB
file, so `make etl` reconstructs every table and view from the CSV in one command.

Run: python -m customer_churn.etl
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Config, project_root


def connect(cfg: Config):
    import duckdb

    db = cfg.data_path("duckdb")
    db.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db))


def run_scripts(cfg: Config, con=None) -> list[str]:
    """Execute the numbered SQL scripts in order, from the repo root."""
    own = con is None
    con = con or connect(cfg)
    executed = []
    root = project_root()
    for name in cfg.raw["sql_scripts"]:
        path = root / cfg.raw["sql_dir"] / name
        sql = path.read_text()
        # read_csv_auto paths in the scripts are repo-relative
        con.execute(f"SET FILE_SEARCH_PATH='{root}'")
        con.execute(sql)
        executed.append(name)
    if own:
        con.commit()
    return executed


def assert_staging_sane(con) -> dict:
    """Refuse to proceed on a truncated or duplicated load."""
    row = con.execute("SELECT * FROM stg_rowcount_check").fetchdf().iloc[0].to_dict()
    if not row["customer_id_is_unique"]:
        raise ValueError("Customer_ID is not unique in staging -- the load is wrong")
    if row["n_rows"] < 6000:
        raise ValueError(f"staging has only {row['n_rows']} rows; expected ~6,418")
    return row


def modelling_frame(cfg: Config, con=None) -> pd.DataFrame:
    con = con or connect(cfg)
    return con.execute("SELECT * FROM vw_churn_data").fetchdf()


def scoring_frame(cfg: Config, con=None) -> pd.DataFrame:
    con = con or connect(cfg)
    return con.execute("SELECT * FROM vw_join_data").fetchdf()


def main() -> None:
    cfg = Config.load()
    con = connect(cfg)
    done = run_scripts(cfg, con)
    stats = assert_staging_sane(con)
    n_model = con.execute("SELECT COUNT(*) FROM vw_churn_data").fetchone()[0]
    n_score = con.execute("SELECT COUNT(*) FROM vw_join_data").fetchone()[0]
    print(f"ran: {', '.join(done)}")
    print(f"staging: {stats['n_rows']} rows, {stats['n_customers']} unique customers")
    print(f"vw_churn_data: {n_model} rows | vw_join_data: {n_score} rows")
    print(f"database: {cfg.data_path('duckdb')}")
    con.close()


if __name__ == "__main__":
    main()
