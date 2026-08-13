"""The DuckDB port must reproduce the SQL Server logic, not merely run."""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from customer_churn.audit import load_raw  # noqa: E402
from customer_churn.config import Config  # noqa: E402
from customer_churn.etl import assert_staging_sane, run_scripts  # noqa: E402


@pytest.fixture(scope="module")
def con():
    cfg = Config.load()
    connection = duckdb.connect(":memory:")
    run_scripts(cfg, connection)
    yield connection
    connection.close()


def test_staging_matches_the_csv(con, raw):
    n = con.execute("SELECT COUNT(*) FROM stg_churn").fetchone()[0]
    assert n == len(raw)


def test_staging_sanity_check_passes(con):
    stats = assert_staging_sane(con)
    assert stats["customer_id_is_unique"]


def test_coalesce_removed_every_null_it_targeted(con):
    """The SQL Server ISNULL() calls became COALESCE(); verify none slipped."""
    cols = ["Value_Deal", "Multiple_Lines", "Internet_Type", "Online_Security",
            "Online_Backup", "Device_Protection_Plan", "Premium_Support",
            "Streaming_TV", "Streaming_Movies", "Streaming_Music", "Unlimited_Data"]
    for c in cols:
        n = con.execute(f"SELECT COUNT(*) FROM prod_churn WHERE {c} IS NULL").fetchone()[0]
        assert n == 0, f"{c} still has nulls after the production build"


def test_null_counts_covers_every_column(con):
    """Drift guard for sql/03_nullcheck.sql.

    The null audit lists its columns explicitly, so adding a column to the source
    would silently leave it unaudited. This fails the build instead.
    """
    source = {r[0] for r in con.execute("DESCRIBE stg_churn").fetchall()}
    audited = {r[0] for r in con.execute("SELECT column_name FROM null_counts").fetchall()}
    assert audited == source, f"unaudited columns: {sorted(source - audited)}"


def test_null_counts_agrees_with_pandas(con, modelling):
    """Cross-check the SQL against an independent implementation.

    The view and pandas share no code, so agreement is evidence the port is right
    rather than evidence it is self-consistent.
    """
    import pandas as pd

    sql_counts = con.execute(
        "SELECT column_name, null_count FROM null_counts ORDER BY column_name"
    ).fetchdf().set_index("column_name")["null_count"]

    raw = pd.read_csv("data/raw/Customer_Data.csv")
    pandas_counts = raw.isna().sum()

    for col, expected in pandas_counts.items():
        assert int(sql_counts[col]) == int(expected), (
            f"{col}: SQL says {sql_counts[col]}, pandas says {expected}"
        )


def test_modelling_view_excludes_outcome_columns(con):
    cols = [r[0] for r in con.execute("DESCRIBE vw_churn_data").fetchall()]
    assert "Churn_Category" not in cols
    assert "Churn_Reason" not in cols


def test_views_partition_the_customers(con, raw):
    a = con.execute("SELECT COUNT(*) FROM vw_churn_data").fetchone()[0]
    b = con.execute("SELECT COUNT(*) FROM vw_join_data").fetchone()[0]
    assert a + b == len(raw), "the two views must partition prod_churn exactly"


def test_churn_rate_by_contract_ranks_month_to_month_first(con):
    top = con.execute("SELECT Contract FROM churn_rate_by_contract LIMIT 1").fetchone()[0]
    assert top == "Month-to-Month"
