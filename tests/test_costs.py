import pytest

from ostinote import costs


def test_costs_day_totals(tmp_path):
    """Summarize token and cost lines from daily memory logs.

    Expected: only `memory-YYYY-MM-DD.log` files with token lines count; totals
    aggregate calls, input, cache, output, and only the cost values actually
    reported by the model engine.
    """
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "memory-2026-06-09.log").write_text(
        "12:00:00 [save] tokens: 100+50cache→20out ($0.000123)\n"
        "12:30:00 [compress] tokens: 200+0cache→40out\n"
        "12:31:00 [hook] not a token line\n",
        encoding="utf-8",
    )
    (logs / "memory-2026-06-10.log").write_text("09:00:00 [hook] no calls today\n", encoding="utf-8")
    (logs / "background.log").write_text("[save] tokens: 9+9cache→9out ($9)\n", encoding="utf-8")

    days = costs.day_totals(str(logs))
    assert [d for d, _ in days] == ["2026-06-09"]  # only daily logs with calls
    totals = days[0][1]
    assert totals["calls"] == 2
    assert totals["input"] == 300
    assert totals["cache"] == 50
    assert totals["output"] == 60
    assert totals["cost"] == pytest.approx(0.000123)  # unreported cost not invented
