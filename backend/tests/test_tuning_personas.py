"""Strategy tuning, copy-trading personas and the strategy chat (no network, no LLM)."""

import asyncio
import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from conftest import make_settings
from wallex_agent import personas, strategy_chat, tuning


def test_tuning_validates_bounds_types_and_cross_rules():
    ok = tuning.validate({"engine_max_positions": "2", "engine_core_enabled": "false", "engine_setups": ["dip_in_uptrend"],
                          "engine_decider": "rules", "engine_max_exposure_pct": 0.5})
    assert ok == {"engine_max_positions": 2, "engine_core_enabled": False, "engine_setups": "dip_in_uptrend",
                  "engine_decider": "rules", "engine_max_exposure_pct": 0.5}
    for bad in ({"engine_max_positions": 9}, {"engine_quote": "TMN"}, {"engine_decider": "magic"},
                {"engine_setups": "moonshot"}, {"engine_drawdown_pause_pct": 0.1, "engine_drawdown_stop_pct": 0.08},
                {"engine_max_exposure_pct": "lots"}):
        with pytest.raises(tuning.TuningError):
            tuning.validate(bad)


def test_tuning_env_roundtrip_feeds_the_engine_settings(tmp_path):
    vals = tuning.validate({"engine_setups": "", "engine_breadth_gate_pct": 15, "engine_final_sweep": False})
    f = tmp_path / "tuning.env"
    f.write_text(tuning.to_env(vals))
    assert tuning.read_env(f) == vals
    import os
    env = dict(line.split("=", 1) for line in f.read_text().splitlines())
    s = make_settings(tmp_path, **{k.lower(): v for k, v in env.items()})
    assert s.engine_setups == "" and s.engine_breadth_gate_pct == 15 and s.engine_final_sweep is False


def test_every_persona_preset_is_valid_and_has_a_profile():
    ids = set()
    for p in personas.PERSONAS:
        tuning.validate(p["preset"])
        assert p["name"] and p["tagline"] and p["bio"] and 1 <= p["risk"] <= 5 and 0 <= p["hue"] < 360
        ids.add(p["id"])
    assert len(ids) == len(personas.PERSONAS)
    assert "strategist" not in personas.session_ids() and "guardian" in personas.session_ids()
    from wallex_agent.engine.runner import valid_run_name
    # live, 2026-10-23: persona-dip_hunter was refused at startup (underscore) and its unit crash-looped
    assert all(valid_run_name(personas.run_name(p)) for p in personas.PERSONAS)
    assert not any(valid_run_name(n) for n in ("live", "shadow", "../x", "a/b", ""))


@pytest.fixture
def srv(tmp_path, monkeypatch):
    from wallex_agent import server

    calls = []

    class Proc:
        returncode = 0
        async def communicate(self):
            return b"", b""

    async def fake_exec(*args, **kw):
        calls.append(args)
        return Proc()
    eng = tmp_path / "data/engine"
    eng.mkdir(parents=True)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server, "ENGINE_MODE_FILE", eng / "mode.env")
    monkeypatch.setattr(server, "STRATEGY_PROFILE_FILE", eng / "profile.env")
    monkeypatch.setattr(server, "TUNING_FILE", eng / "tuning.env")
    monkeypatch.setattr(server, "COPY_FILE", eng / "copy.json")
    monkeypatch.setattr(server, "CONTROL_FILE", eng / "control.json")
    monkeypatch.setattr(server.shutil, "which", lambda name: "/bin/systemctl")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    (eng / "mode.env").write_text("ENGINE_RUN_ARGS=--mode shadow\n")
    return NS(server=server, c=TestClient(server.app), calls=calls, eng=eng, tmp=tmp_path)


def test_tuning_endpoint_saves_restarts_and_guards_live(srv):
    c = srv.c
    r = c.get("/api/strategy/tuning").json()
    assert {p["key"] for p in r["params"]} >= {"engine_max_exposure_pct", "engine_setups"} and not r["live"]
    assert c.post("/api/strategy/tuning", json={"values": {"engine_max_positions": 99}}).status_code == 400
    assert c.post("/api/strategy/tuning", json={"values": {"engine_max_positions": 1}}).json()["ok"]
    assert "ENGINE_MAX_POSITIONS=1" in (srv.eng / "tuning.env").read_text()
    assert srv.calls[-1] == ("systemctl", "restart", "wallex-engine")
    c.post("/api/strategy/tuning", json={"values": {"engine_breadth_gate_pct": 20}})
    assert tuning.read_env(srv.eng / "tuning.env") == {"engine_max_positions": 1, "engine_breadth_gate_pct": 20.0}
    (srv.eng / "mode.env").write_text("ENGINE_RUN_ARGS=--mode live --confirm-live\n")   # real money: confirm first
    n = len(srv.calls)
    assert c.post("/api/strategy/tuning", json={"values": {"engine_max_positions": 2}}).status_code == 400
    assert len(srv.calls) == n
    assert c.post("/api/strategy/tuning", json={"reset": True, "confirm": "LIVE"}).json()["ok"]
    assert not (srv.eng / "tuning.env").exists()


def test_copy_persona_applies_its_preset_and_stop_copy_reverts(srv):
    c = srv.c
    run = srv.eng / "persona-dip_hunter"
    run.mkdir()
    (run / "equity.csv").write_text("time,equity_tmn,cash_tmn,exposure_tmn,drawdown_pct\n100,5000000,0,0,0\n"
                                    "400,4900000,0,0,2\n700,5100000,0,0,0\n")
    (run / "trades.csv").write_text("trade_id,symbol,setup,entry_time,entry_price,qty,cost_tmn,exit_time,exit_price,"
                                    "proceeds_tmn,pnl_tmn,pnl_pct,exit_reason,ai_confidence,entry_reasoning\n"
                                    "T1,BTCUSDT,dip_in_uptrend,a,1,1,1,b,1,1,12000,0.8,take_profit,,r\n")
    (run / "state.json").write_text(json.dumps({"positions": {}, "core": {"status": "holding"}}))
    r = c.get("/api/personas").json()
    hunter = next(p for p in r["personas"] if p["id"] == "dip_hunter")
    assert hunter["return_pct"] == 2.0 and hunter["max_drawdown_pct"] == 2.0 and hunter["trades"] == 1
    assert hunter["win_rate_pct"] == 100 and hunter["active"] and len(hunter["curve"]) == 3
    assert c.post("/api/personas/nobody/copy", json={}).status_code == 404
    assert c.post("/api/personas/dip_hunter/copy", json={}).json()["copying"]["id"] == "dip_hunter"
    over = tuning.read_env(srv.eng / "tuning.env")
    assert over["engine_max_positions"] == 3 and over["engine_decider"] == "rules"
    assert "ENGINE_QUOTE=USDT" in (srv.eng / "profile.env").read_text()   # personas sit on the strategy profile
    c.post("/api/strategy/tuning", json={"values": {"engine_max_positions": 2}})
    assert c.get("/api/personas").json()["copying"]["modified"]
    assert c.post("/api/personas/stop-copy", json={}).json()["ok"]
    assert not (srv.eng / "tuning.env").exists() and c.get("/api/personas").json()["copying"] is None


def test_copy_and_stop_copy_keep_the_persons_challenge_window(srv):
    """Copying a persona swaps the trading style, not the end of the challenge the person set."""
    c = srv.c
    end = "2099-01-01T12:00:00+03:30"
    assert c.post("/api/strategy/tuning", json={"values": {"engine_end_time": end, "engine_max_positions": 1}}).json()["ok"]
    c.post("/api/personas/cautious/copy", json={})
    over = tuning.read_env(srv.eng / "tuning.env")
    assert over["engine_end_time"] == "2099-01-01T08:30:00Z" and over["engine_max_positions"] == 1   # cautious: 1
    assert over["engine_breadth_gate_pct"] == 40.0
    c.post("/api/personas/dip_hunter/copy", json={})
    over = tuning.read_env(srv.eng / "tuning.env")
    assert over["engine_end_time"] == "2099-01-01T08:30:00Z" and over["engine_max_positions"] == 3
    assert "engine_drawdown_stop_pct" not in over                                  # the previous persona is gone
    c.post("/api/personas/stop-copy", json={})
    assert tuning.read_env(srv.eng / "tuning.env") == {"engine_end_time": "2099-01-01T08:30:00Z"}


def test_a_saved_end_time_survives_once_it_is_near(srv):
    """Only a newly typed end time must be an hour ahead; one saved earlier is not dropped when it gets close."""
    import time as _t
    from datetime import datetime, timezone
    soon = datetime.fromtimestamp(_t.time() + 600, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (srv.eng / "tuning.env").write_text(f"ENGINE_END_TIME={soon}\n")
    assert tuning.read_env(srv.eng / "tuning.env") == {"engine_end_time": soon}
    assert srv.c.post("/api/strategy/tuning", json={"values": {"engine_max_positions": 2}}).json()["ok"]
    assert tuning.read_env(srv.eng / "tuning.env")["engine_end_time"] == soon
    with pytest.raises(tuning.TuningError):
        tuning.validate({"engine_end_time": soon})


def test_drawdown_pause_is_checked_against_the_running_stop(srv):
    """Pause 0.10 alone is in bounds, but not above a stop of 0.08 already saved."""
    c = srv.c
    assert c.post("/api/strategy/tuning", json={"values": {"engine_drawdown_stop_pct": 0.08}}).json()["ok"]
    r = c.post("/api/strategy/tuning", json={"values": {"engine_drawdown_pause_pct": 0.10}})
    assert r.status_code == 400 and "pause" in r.json()["detail"]


def test_chat_proposal_is_validated_and_only_keeps_real_changes():
    eff = {"engine_max_positions": 2, "engine_breadth_gate_pct": 25.0, "engine_drawdown_pause_pct": 0.06,
           "engine_drawdown_stop_pct": 0.12}
    p = strategy_chat._proposal({"changes": {"max_positions": 3, "engine_breadth_gate_pct": 25, "engine_quote": "TMN",
                                             "engine_min_score": 7}, "reasoning": "x"}, eff)
    assert p["changes"] == {"engine_max_positions": 3} and len(p["dropped"]) == 2
    assert strategy_chat._proposal({"changes": {"engine_max_positions": 2}}, eff) is None
    bad = strategy_chat._proposal({"changes": {"engine_drawdown_pause_pct": 0.13}}, eff)
    assert bad["changes"] == {} and bad["dropped"]


async def test_chat_reads_a_tool_call_proposal(tmp_path, monkeypatch):
    call = NS(function=NS(name="propose_changes",
                          arguments=json.dumps({"changes": {"engine_breadth_gate_pct": 15}, "reasoning": "بیشتر"})))

    class FakeLLM:
        def __init__(self, **kw):
            self.chat = NS(completions=NS(create=self.create))

        async def create(self, **kw):
            assert kw["tools"][0]["function"]["name"] == "propose_changes"
            assert "CURRENT PARAMETERS" in kw["messages"][0]["content"]
            return NS(choices=[NS(message=NS(content="دروازه را باز کنیم.", tool_calls=[call]))])
    monkeypatch.setattr(strategy_chat, "AsyncOpenAI", FakeLLM)
    r = await strategy_chat.ask(make_settings(tmp_path), [{"role": "user", "content": "more trades"}],
                                {"engine_breadth_gate_pct": 25.0}, {}, None)
    assert r["reply"] and r["proposal"]["changes"] == {"engine_breadth_gate_pct": 15.0}


def test_end_time_is_tunable_in_tehran_time_and_never_in_the_past():
    assert tuning.validate({"engine_end_time": "2099-10-26T23:59:00+03:30"}) == {"engine_end_time": "2099-10-26T20:29:00Z"}
    for bad in ("2020-01-01T00:00:00+03:30", "2099-10-26T23:59:00", "monday"):
        with pytest.raises(tuning.TuningError):
            tuning.validate({"engine_end_time": bad})
    assert [p["key"] for p in tuning.PARAMS if p["group"] == "time"][:3] == [
        "engine_end_time", "engine_no_entry_before_end_min", "engine_flatten_before_end_min"]


def test_assistant_applies_changes_itself_saves_the_chat_and_can_undo(srv, monkeypatch):
    calls = iter([
        {"reply": "انجام شد.", "proposal": {"changes": {"engine_max_positions": 1, "engine_risk_per_trade_pct": 0.005},
                                            "reasoning": "", "dropped": []}, "apply": True},
        {"reply": "پیشنهاد:", "proposal": {"changes": {"engine_breadth_gate_pct": 30.0}, "reasoning": "", "dropped": []},
         "apply": False},
        {"reply": "انجام شد.", "proposal": {"changes": {"engine_max_positions": 3}, "reasoning": "", "dropped": []},
         "apply": True},
    ])
    seen = {}

    async def fake_ask(s, history, effective, engine, copying, allow_apply=False):
        seen["allow"], seen["history"] = allow_apply, [m["content"] for m in history]
        return next(calls)
    monkeypatch.setattr(strategy_chat, "ask", fake_ask)
    monkeypatch.setattr(srv.server.settings, "llm_api_key", "k")
    monkeypatch.setattr(srv.server, "STRATEGY_CHATS", srv.tmp / "data/strategy_chats")
    (srv.eng / "tuning.env").write_text("ENGINE_MAX_POSITIONS=2\n")
    c = srv.c
    r = c.post("/api/strategy/chat", json={"message": "ریسک را نصف کن و یک پوزیشن"}).json()
    sid = r["session_id"]
    assert seen["allow"] and r["message"]["applied"]["changes"]["engine_max_positions"] == 1
    assert tuning.read_env(srv.eng / "tuning.env") == {"engine_max_positions": 1, "engine_risk_per_trade_pct": 0.005}
    assert srv.calls[-1] == ("systemctl", "restart", "wallex-engine")
    r2 = c.post("/api/strategy/chat", json={"message": "دروازه چطور؟", "session_id": sid, "allow_apply": False}).json()
    assert "applied" not in r2["message"] and seen["history"] == ["ریسک را نصف کن و یک پوزیشن", "انجام شد.", "دروازه چطور؟"]
    rows = c.get("/api/strategy/chats").json()
    assert rows[0]["id"] == sid and rows[0]["turns"] == 2
    assert c.post(f"/api/strategy/chats/{sid}/undo", json={"index": 3}).status_code == 400   # that one only proposed
    assert c.post(f"/api/strategy/chats/{sid}/undo", json={"index": 1}).json()["ok"]
    assert tuning.read_env(srv.eng / "tuning.env") == {"engine_max_positions": 2}
    assert c.get(f"/api/strategy/chats/{sid}").json()["messages"][1]["undone"]
    (srv.eng / "mode.env").write_text("ENGINE_RUN_ARGS=--mode live --confirm-live\n")       # live: needs the confirmation
    r3 = c.post("/api/strategy/chat", json={"message": "و حالا؟", "session_id": sid}).json()
    assert "applied" not in r3["message"] and "LIVE" in r3["message"]["apply_error"]   # nothing changed without it
    assert c.delete(f"/api/strategy/chats/{sid}").json()["ok"] and c.get("/api/strategy/chats").json() == []


def test_engine_switch_pauses_freely_but_turning_on_live_needs_confirmation(srv):
    c = srv.c
    assert c.post("/api/engine/switch", json={"on": False}).json() == {"ok": True, "on": False}
    (srv.eng / "mode.env").write_text("ENGINE_RUN_ARGS=--mode live --confirm-live\n")
    c.post("/api/engine/switch", json={"on": False})
    assert c.post("/api/engine/switch", json={"on": True}).status_code == 400 and srv.server.engine_paused()
    assert c.post("/api/engine/switch", json={"on": True, "confirm": "LIVE"}).json()["on"]


def test_undo_reverts_only_what_that_message_changed(srv, monkeypatch):
    """Undoing an assistant change must not wipe what was set after it (by hand or by a later message)."""
    calls = iter([
        {"reply": "ok", "proposal": {"changes": {"engine_max_positions": 1}, "reasoning": "", "dropped": []}, "apply": True},
        {"reply": "ok", "proposal": {"changes": {"engine_risk_per_trade_pct": 0.005}, "reasoning": "", "dropped": []},
         "apply": True},
    ])

    async def fake_ask(*a, **kw):
        return next(calls)
    monkeypatch.setattr(strategy_chat, "ask", fake_ask)
    monkeypatch.setattr(srv.server.settings, "llm_api_key", "k")
    monkeypatch.setattr(srv.server, "STRATEGY_CHATS", srv.tmp / "data/strategy_chats")
    (srv.eng / "tuning.env").write_text("ENGINE_MAX_POSITIONS=2\n")
    c = srv.c
    sid = c.post("/api/strategy/chat", json={"message": "یک پوزیشن"}).json()["session_id"]
    c.post("/api/strategy/tuning", json={"values": {"engine_breadth_gate_pct": 30}})        # by hand, afterwards
    c.post("/api/strategy/chat", json={"message": "ریسک نصف", "session_id": sid})
    assert c.post(f"/api/strategy/chats/{sid}/undo", json={"index": 1}).json()["ok"]
    assert tuning.read_env(srv.eng / "tuning.env") == {"engine_max_positions": 2, "engine_breadth_gate_pct": 30.0,
                                                        "engine_risk_per_trade_pct": 0.005}
    assert c.post(f"/api/strategy/chats/{sid}/undo", json={"index": 1}).status_code == 400   # already undone
    assert c.post(f"/api/strategy/chats/{sid}/undo", json={"index": 3}).json()["ok"]
    assert tuning.read_env(srv.eng / "tuning.env") == {"engine_max_positions": 2, "engine_breadth_gate_pct": 30.0}


def test_live_traders_are_listed_added_and_removed(srv, monkeypatch):
    from wallex_agent import traders
    monkeypatch.setattr(srv.server.traders_mod, "DIR", srv.eng / "traders")
    c = srv.c
    assert c.post("/api/traders", json={"address": "nope"}).status_code == 400
    addr = "0x" + "ab" * 20
    assert c.post("/api/traders", json={"address": addr.upper().replace("0X", "0x"), "name": "Famous"}).json()["ok"]
    run = srv.eng / traders.run_name(addr)
    run.mkdir()
    (run / "equity.csv").write_text("time,equity_tmn,cash_tmn,exposure_tmn,drawdown_pct\n100,5000000,0,0,0\n200,5050000,0,0,0\n")
    (run / "trader.json").write_text(json.dumps({"account_usd": 2e6, "positions": [{"coin": "BTC", "side": "long",
                                                 "usd": 1e6, "size": 10, "entry": 1, "upnl": 0}], "board": {"pnl_month": 5}}))
    r = c.get("/api/personas").json()["traders"]
    assert len(r) == 1 and r[0]["name"] == "Famous" and r[0]["source"] == "pinned" and r[0]["return_pct"] == 1.0
    assert r[0]["positions"][0]["coin"] == "BTC" and r[0]["account_usd"] == 2e6
    state = {"cash_usdt": 10.0, "holdings": {"BTCUSDT": 0.1}, "positions": [{"coin": "BTC", "side": "long"}]}
    (run / "state.json").write_text(json.dumps(state))                  # a follower's state: positions is a list
    assert c.get("/api/personas").json()["traders"][0]["open_positions"] == ["BTCUSDT"]
    (run / "state.json").write_text(json.dumps({**state, "positions": None}))   # before its first poll
    assert c.get("/api/personas").json()["traders"][0]["open_positions"] == ["BTCUSDT"]
    assert c.delete(f"/api/traders/{addr}").json()["ok"] and c.get("/api/traders").json()["traders"] == []


def test_copying_a_live_trader_points_the_engine_at_them_and_stop_copy_clears_it(srv, monkeypatch):
    from wallex_agent import traders
    monkeypatch.setattr(srv.server.traders_mod, "DIR", srv.eng / "traders")
    monkeypatch.setattr(srv.server, "STRATEGY_ENV", srv.tmp / "strategy.env")
    (srv.tmp / "strategy.env").write_text("ENGINE_QUOTE=USDT\n")
    c = srv.c
    addr = "0x" + "ef" * 20
    assert c.post(f"/api/traders/{addr}/copy", json={}).status_code == 404               # follow it first
    c.post("/api/traders", json={"address": addr, "name": "Whale"})
    c.post("/api/strategy/tuning", json={"values": {"engine_end_time": "2099-01-01T00:00:00Z", "engine_max_positions": 1}})
    (srv.eng / "mode.env").write_text("ENGINE_RUN_ARGS=--mode live --confirm-live\n")
    assert c.post(f"/api/traders/{addr}/copy", json={"pct": 0.4}).status_code == 400     # real money: confirm
    r = c.post(f"/api/traders/{addr}/copy", json={"pct": 0.4, "confirm": "LIVE"}).json()
    assert r["copying"]["kind"] == "trader" and r["copying"]["name"] == "Whale"
    over = tuning.read_env(srv.eng / "tuning.env")
    assert over == {"engine_end_time": "2099-01-01T00:00:00Z", "engine_follow_address": addr, "engine_follow_pct": 0.4}
    assert srv.calls[-1] == ("systemctl", "restart", "wallex-engine")
    assert c.post(f"/api/traders/{addr}/copy", json={"pct": 3, "confirm": "LIVE"}).status_code == 400
    assert c.post("/api/personas/stop-copy", json={"confirm": "LIVE"}).json()["ok"]
    assert tuning.read_env(srv.eng / "tuning.env") == {"engine_end_time": "2099-01-01T00:00:00Z"}
    assert c.get("/api/personas").json()["copying"] is None
    assert traders.run_name(addr) == "trader-efefefef"
