import pytest
import pandas as pd
from backend.roster_verification import compute_player_usage_profile
import backend.roster_verification as rv
from backend.scheme_insights import compute_scheme_insights

BANNED_WORDS = ["projected", "expected", "edge", "advantage", "exploit", "pick"]

def test_usage_profile_correct_medians():
    # 1. Bijan Robinson 2026 (weeks 1-5)
    bijan = compute_player_usage_profile("Bijan Robinson", "ATL", season=2026, up_to_week=5)
    assert bijan is not None, "Bijan Robinson usage profile should not be None"
    assert bijan["position"] == "RB"
    assert bijan["n_games"] == 5
    assert bijan["small_sample"] is False
    assert bijan["raw_medians"]["snap_share"] == 72.0
    assert bijan["raw_medians"]["carries"] == 19.0
    assert bijan["raw_medians"]["target_share"] == 12.9
    
    assert bijan["lines"] == [
        {"label": "Snap share", "value": "72%"},
        {"label": "Carries", "value": "19"},
        {"label": "Target share", "value": "12.9%"},
    ]

    # 2. Drake London 2026 (weeks 1-5)
    london = compute_player_usage_profile("Drake London", "ATL", season=2026, up_to_week=5)
    assert london is not None, "Drake London usage profile should not be None"
    assert london["position"] == "WR"
    assert london["n_games"] == 5
    assert london["small_sample"] is False
    assert london["raw_medians"]["snap_share"] == 93.0
    assert london["raw_medians"]["target_share"] == 30.0
    assert london["raw_medians"]["wopr"] == 0.81

    assert london["lines"] == [
        {"label": "Snap share", "value": "93%"},
        {"label": "Target share", "value": "30.0%"},
        {"label": "WOPR", "value": "0.81"},
    ]


def test_inactive_weeks_excluded(monkeypatch):
    # Mock usage df with 4 games, where 1 game has offense_snaps == 0 (inactive)
    mock_df = pd.DataFrame([
        {"season": 2026, "week": 1, "team": "TEST", "player_display_name": "Test Player", "position": "WR", "offense_snaps": 50, "offense_pct": 0.80, "target_share": 0.20, "wopr": 0.50},
        {"season": 2026, "week": 2, "team": "TEST", "player_display_name": "Test Player", "position": "WR", "offense_snaps": 60, "offense_pct": 0.90, "target_share": 0.30, "wopr": 0.70},
        {"season": 2026, "week": 3, "team": "TEST", "player_display_name": "Test Player", "position": "WR", "offense_snaps": 0, "offense_pct": 0.00, "target_share": 0.00, "wopr": 0.00},  # Inactive
        {"season": 2026, "week": 4, "team": "TEST", "player_display_name": "Test Player", "position": "WR", "offense_snaps": 70, "offense_pct": 1.00, "target_share": 0.40, "wopr": 0.90},
    ])
    monkeypatch.setattr(rv, "_CACHED_USAGE_DF", mock_df)

    res = compute_player_usage_profile("Test Player", "TEST", season=2026)
    assert res is not None
    # Must only count the 3 active games
    assert res["n_games"] == 3
    assert res["small_sample"] is True
    # Medians over weeks 1, 2, 4: snap share = 90%, target share = 30.0%, wopr = 0.70
    assert res["raw_medians"]["snap_share"] == 90.0
    assert res["raw_medians"]["target_share"] == 30.0
    assert res["raw_medians"]["wopr"] == 0.70


def test_small_sample_flag():
    # Bijan with up_to_week=3 gives exactly 3 games
    res = compute_player_usage_profile("Bijan Robinson", "ATL", season=2026, up_to_week=3)
    assert res is not None
    assert res["n_games"] == 3
    assert res["small_sample"] is True


def test_missing_player_returns_none():
    res = compute_player_usage_profile("Unknown Nonexistent Player", "ATL", season=2026)
    assert res is None


def test_banned_language_check():
    # Verify no banned words in generated text for Bijan and London
    for name in ["Bijan Robinson", "Drake London"]:
        res = compute_player_usage_profile(name, "ATL", season=2026, up_to_week=5)
        assert res is not None
        for line in res["lines"]:
            label = line["label"].lower()
            val = str(line["value"]).lower()
            for banned in BANNED_WORDS:
                assert banned not in label, f"Banned word '{banned}' found in label '{line['label']}'"
                assert banned not in val, f"Banned word '{banned}' found in value '{line['value']}'"


def test_scheme_insights_player_payload_rendering_check():
    insights = compute_scheme_insights("ATL", "CHI", season=2026, week=5)
    assert "player_usage_profiles" in insights
    profs = insights["player_usage_profiles"]
    assert len(profs) > 0
    for p in profs:
        assert "usage_profile" in p
        if p.get("player_display_name") in ["Bijan Robinson", "Drake London"]:
            assert p["usage_profile"] is not None
            assert "lines" in p["usage_profile"]
            assert len(p["usage_profile"]["lines"]) == 3
            assert p["usage_profile"]["n_games"] >= 4


def test_wopr_median_exact_rounding():
    # Drake London 2026 weeks 1-5 has raw WOPR values:
    # [0.515789, 0.343193, 1.107879, 0.805385, 0.832377]
    # The sorted values are: [0.343193, 0.515789, 0.805385, 0.832377, 1.107879]
    # The middle element (median) is 0.8053846153846154.
    # Verify that this value exceeds 0.805 and unambiguously rounds to 0.81.
    res = compute_player_usage_profile("Drake London", "ATL", season=2026, up_to_week=5)
    assert res is not None
    raw_wopr = res["raw_medians"]["wopr"]
    assert raw_wopr == 0.81
    wopr_line = [l for l in res["lines"] if l["label"] == "WOPR"][0]
    assert wopr_line["value"] == "0.81"


def test_multi_game_sample_2024():
    # 2024 full season (17 games) for Bijan Robinson and Drake London
    bijan_24 = compute_player_usage_profile("Bijan Robinson", "ATL", season=2024)
    assert bijan_24 is not None
    assert bijan_24["n_games"] == 17
    assert bijan_24["small_sample"] is False
    assert bijan_24["position"] == "RB"
    assert bijan_24["raw_medians"]["snap_share"] == 75.0
    assert bijan_24["raw_medians"]["carries"] == 18.0
    assert bijan_24["raw_medians"]["target_share"] == 13.8

    london_24 = compute_player_usage_profile("Drake London", "ATL", season=2024)
    assert london_24 is not None
    assert london_24["n_games"] == 17
    assert london_24["small_sample"] is False
    assert london_24["position"] == "WR"
    assert london_24["raw_medians"]["snap_share"] == 98.0
    assert london_24["raw_medians"]["target_share"] == 27.0
    assert london_24["raw_medians"]["wopr"] == 0.63


def test_missing_cache_file_logs_warning_and_returns_none(monkeypatch, caplog):
    # Simulate missing cache path and ensure a visible warning is logged
    monkeypatch.setattr(rv, "_CACHED_USAGE_DF", None)
    monkeypatch.setattr(rv, "USAGE_CACHE_PATH", "nonexistent/path/player_usage_cache.parquet")

    with caplog.at_level("WARNING"):
        res = compute_player_usage_profile("Bijan Robinson", "ATL", season=2026)

    assert res is None
    assert any("Usage cache not found at nonexistent/path" in record.message for record in caplog.records)


