"""FastAPI server: SSE chat with the agent + dashboard data endpoints + static UI."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import shutil
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import tools as agent_tools
from .agent.context import AppContext
from .agent.loop import Agent
from .brokers import BrokerError, LiveSpotBroker, PaperFuturesBroker, PaperSpotBroker
from .config import ROOT, get_settings
from .market import analytics, sentiment
from .news import web
from . import personas as persona_mod, strategy_chat, traders as traders_mod, tuning
from .wallex import WallexError

settings = get_settings()
ENGINE_MODE_FILE = ROOT / "data" / "engine" / "mode.env"


def requested_engine_mode() -> str:
    """The engine's configured mode: the UI switch (data/engine/mode.env) or ENGINE_RUN_ARGS from .env."""
    try:
        args = ENGINE_MODE_FILE.read_text()
    except OSError:
        args = os.environ.get("ENGINE_RUN_ARGS", "")
    return "live" if "--mode live" in args else "shadow"


# One mode for the whole app: when the engine trades for real, so do the chat assistant's spot trades
settings.spot_mode = "live" if requested_engine_mode() == "live" and settings.wallex_api_key else "paper"
ctx = AppContext.build(settings)
agent = Agent(ctx)
futures_events: collections.deque = collections.deque(maxlen=50)


async def _ticker():
    while True:
        if isinstance(ctx.futures, PaperFuturesBroker):
            try:
                for ev in await ctx.futures.tick():
                    futures_events.appendleft({**ev, "seen_at": time.time()})
            except Exception:
                pass
        try:
            await ctx.alerts.check()
        except Exception:
            pass
        try:
            await _run_strategy_schedule()
        except Exception:
            pass
        await asyncio.sleep(10)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_ticker())
    yield
    task.cancel()
    await ctx.client.aclose()


app = FastAPI(title="Wallex Agent", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def sse(gen):
    async def stream():
        try:
            async for ev in gen:
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def guarded(coro):
    try:
        return await coro
    except (WallexError, BrokerError) as e:
        raise HTTPException(502, str(e))


# ------------------------------------------------------------------- agent
class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


@app.post("/api/chat")
async def chat(body: ChatIn):
    session = agent.session(body.session_id)

    async def gen():
        yield {"type": "session", "session_id": session.id}
        try:
            async for ev in agent.chat(session, body.message):
                yield ev
        finally:
            agent.save(session)
    return sse(gen())


class ActionIn(BaseModel):
    approve: bool


@app.post("/api/actions/{action_id}")
async def decide_action(action_id: str, body: ActionIn):
    try:
        action = await agent.execute_action(action_id, body.approve)
    except KeyError:
        raise HTTPException(404, "unknown action")

    async def gen():
        yield {"type": "action_update", "action": action.public()}
        try:
            async for ev in agent.after_action(action):
                yield ev
        finally:
            if action.session_id in agent.sessions:
                agent.save(agent.sessions[action.session_id])
    return sse(gen())


@app.get("/api/actions")
async def list_actions(session_id: str | None = None):
    return [a.public() for a in agent.pending.values() if not session_id or a.session_id == session_id]


@app.post("/api/sessions/{session_id}/reset")
async def reset_session(session_id: str):
    """Leaves the conversation (it stays saved and listed); drops only the in-memory copy."""
    agent.sessions.pop(session_id, None)
    return {"ok": True}


@app.get("/api/sessions")
async def list_sessions():
    return agent.store.list()


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    d = agent.store.load(session_id)
    if d is None:
        raise HTTPException(404, "unknown session")
    return {"id": d["id"], "title": d.get("title"), "updated": d.get("updated"), "transcript": d.get("transcript", [])}


class TranscriptIn(BaseModel):
    transcript: list


@app.put("/api/sessions/{session_id}")
async def save_transcript(session_id: str, body: TranscriptIn):
    if not agent.store.save_transcript(session_id, body.transcript):
        raise HTTPException(404, "unknown session")
    return {"ok": True}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    agent.sessions.pop(session_id, None)
    return {"ok": agent.store.delete(session_id)}


@app.get("/api/config")
async def config():
    return {"spot_mode": settings.spot_mode, "futures_mode": settings.futures_mode,
            "model": settings.llm_model, "require_confirmation": settings.require_confirmation,
            "has_wallex_key": bool(settings.wallex_api_key), "llm_configured": bool(settings.llm_api_key),
            "max_leverage": settings.max_leverage, "tools": sorted(agent_tools.TOOLS),
            "engine_mode": (engine_status() or {}).get("mode") or requested_engine_mode()}


# --------------------------------------------------------------- dashboard
@app.get("/api/markets")
async def markets(quote: str | None = None, sort: str = "volume", limit: int = 30, search: str | None = None):
    return await guarded(analytics.market_overview(ctx.client, quote, sort, limit, search))


@app.get("/api/movers")
async def movers(hours: int = 4, quote: str = "USDT", direction: str = "gainers", limit: int = 10):
    return await guarded(analytics.top_movers(ctx.client, hours, quote, limit, direction))


async def known_symbol(symbol: str) -> str:
    sym = symbol.upper()
    if sym not in await guarded(ctx.client.markets()):
        raise HTTPException(404, f"unknown market {sym}")
    return sym


@app.get("/api/depth/{symbol}")
async def depth(symbol: str, levels: int = 12):
    sym = await known_symbol(symbol)
    return await guarded(analytics.order_book_analysis(ctx.client, sym, max(1, min(levels, 50))))


@app.get("/api/trades/{symbol}")
async def market_trades(symbol: str, limit: int = 30):
    """The market's latest public trades (newest first) for the trade tape."""
    sym = await known_symbol(symbol)
    rows = await guarded(ctx.client.trades(sym))
    out = [{"time": t.get("timestamp"), "price": float(t["price"]), "qty": float(t["quantity"]),
            "side": "buy" if t.get("isBuyOrder") else "sell"} for t in rows[: max(1, min(limit, 100))]]
    return {"symbol": sym, "trades": out}


@app.get("/api/candles/{symbol}")
async def candles(symbol: str, resolution: str = "60", bars: int = 150):
    if resolution not in analytics.RES_SECONDS:
        raise HTTPException(400, f"resolution must be one of {sorted(analytics.RES_SECONDS)}")
    sym = await known_symbol(symbol)
    bars = max(1, min(bars, 1000))
    now = int(time.time())
    return await guarded(ctx.client.candles(sym, resolution, now - analytics.RES_SECONDS[resolution] * bars, now))


@app.get("/api/sentiment")
async def sentiment_ep(quote: str = "USDT"):
    try:
        glob = await sentiment.global_fear_greed(30)
    except Exception as e:
        glob = {"error": str(e)}
    return {"global": glob, "wallex_pulse": await guarded(sentiment.wallex_pulse(ctx.client, quote))}


@app.get("/api/news")
async def news(query: str | None = None, limit: int = 15):
    return await web.crypto_news(query, limit)


@app.get("/api/portfolio")
async def portfolio():
    out: dict = {"spot_mode": ctx.spot.mode, "futures_mode": ctx.futures.mode, "events": list(futures_events)[:10]}
    try:
        out["spot"] = await agent_tools.spot_balances(ctx, agent_tools.NoArgs())
        out["spot_orders"] = await ctx.spot.open_orders()
    except (WallexError, BrokerError) as e:
        out["spot_error"] = str(e)
    try:
        out["futures"] = await ctx.futures.account()
        out["positions"] = await ctx.futures.positions()
        out["futures_orders"] = await ctx.futures.open_orders()
    except (WallexError, BrokerError) as e:
        out["futures_error"] = str(e)
    return out


ENGINE_EVENTS = {"sweep", "core", "core_decision", "regime", "decision", "risk_block", "entry_order", "entry", "entry_cancelled", "exit", "review",
                 "circuit_breaker", "error", "start", "stop", "trader"}


STRATEGY_RUN = "strategy"   # deploy/wallex-strategy.service: the USDT strategy in shadow beside the main engine


def engine_dir(mode: str | None = None) -> Path | None:
    """The journal directory of the engine run in `mode` (default: the app's mode; else whichever exists).
    A live engine and a shadow session can run side by side, so the newest file must not decide."""
    order = [mode] if mode else [requested_engine_mode(), "shadow", "live"]
    return next((d for d in (ROOT / "data" / "engine" / m for m in order) if (d / "state.json").exists()), None)


def engine_status(mode: str | None = None) -> dict | None:
    """The autonomous engine's state, read from its journal (it runs as a separate process)."""
    d = engine_dir(mode)
    if d is None:
        return None
    state = json.loads((d / "state.json").read_text())
    eq_rows = (d / "equity.csv").read_text().strip().splitlines()[1:] if (d / "equity.csv").exists() else []
    last = eq_rows[-1].split(",") if eq_rows else None
    events = []
    if (d / "decisions.jsonl").exists():
        with (d / "decisions.jsonl").open("rb") as f:   # only the tail: the journal grows all challenge long
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 200_000))
            lines = f.read().decode(errors="ignore").splitlines()[1:]
        for line in reversed(lines):
            with contextlib.suppress(ValueError):
                e = json.loads(line)
                if e.get("event") in ENGINE_EVENTS:
                    dec = e.get("decision") or {}
                    events.append({k: v for k, v in {
                        "time": e["time"], "event": e["event"], "symbol": e.get("symbol"), "setup": e.get("setup"),
                        "action": e.get("action"), "regime": e.get("regime"), "source": e.get("source"),
                        "reason": e.get("reason") or dec.get("reasoning") or e.get("error"),
                        "pnl_pct": e.get("pnl_pct"), "price": e.get("price")}.items() if v is not None})
            if len(events) >= 30:
                break
    trades = []
    if (d / "trades.csv").exists():
        with (d / "trades.csv").open() as f:
            trades = list(csv.DictReader(f))
    pnls = [float(t["pnl_tmn"] or 0) for t in trades]
    stats = {"trades": len(trades), "wins": sum(1 for x in pnls if x > 0), "realized_pnl_tmn": round(sum(pnls)),
             "win_rate_pct": round(100 * sum(1 for x in pnls if x > 0) / len(pnls), 1) if pnls else None}
    pending = [{"symbol": o.get("symbol"), "setup": o.get("setup"), "limit": o.get("limit"),
                "amount_tmn": o.get("amount_tmn"), "placed_at": o.get("placed_at"),
                "expires_at": o.get("expires_at")} for o in state.get("pending", {}).values()]
    recent = [{"id": t["trade_id"], "symbol": t["symbol"], "setup": t["setup"], "entry_time": t["entry_time"],
               "exit_time": t["exit_time"], "entry_price": float(t["entry_price"] or 0),
               "exit_price": float(t["exit_price"] or 0), "pnl_tmn": float(t["pnl_tmn"] or 0),
               "pnl_pct": float(t["pnl_pct"] or 0), "reason": t["exit_reason"]} for t in reversed(trades[-20:])]
    return {"mode": d.name, "updated": (d / "state.json").stat().st_mtime, "started_ts": _last_start(d).get("ts"),
            "capital_tmn": settings.engine_capital_tmn,
            "equity_tmn": float(last[1]) if last else None, "cash_tmn": float(last[2]) if last else None,
            "drawdown_pct": float(last[4]) if last else None,
            "regime": state.get("regime"), "regime_reason": state.get("regime_reason"),
            "halted": state.get("halted"), "pause_until": state.get("pause_until"),
            "positions": list(state.get("positions", {}).values()), "pending": pending,
            "trades": recent, "stats": stats, "events": events, "core": state.get("core"),
            "follow": state.get("follow"),
            "exposure_tmn": float(last[3]) if last else None}


MODE_ARGS = {"live": "--mode live --confirm-live", "shadow": "--mode shadow"}
ENGINE_SERVICE = "wallex-engine"


class EngineModeIn(BaseModel):
    mode: Literal["live", "shadow"]
    # switching to real money must be an explicit, deliberate request, not a stray POST
    confirm: str = ""


@app.post("/api/engine/mode")
async def set_engine_mode(body: EngineModeIn):
    """Switch the autonomous engine between shadow and live: write data/engine/mode.env (read by the
    systemd unit after .env) and restart the engine service. Its state resumes per mode."""
    if body.mode == "live" and body.confirm != "LIVE":
        raise HTTPException(400, "switching to live needs confirm=LIVE")
    _need_systemd()
    _write_mode(body.mode)
    await _restart_engine()
    return {"ok": True, "mode": body.mode}


def _need_systemd() -> None:
    if not shutil.which("systemctl"):
        raise HTTPException(501, "the engine is not running as a systemd service here")


def _write_mode(mode: str) -> None:
    ENGINE_MODE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENGINE_MODE_FILE.write_text(f"ENGINE_RUN_ARGS={MODE_ARGS[mode]}\n")
    # the chat assistant follows the engine: real spot orders in live, paper in shadow
    settings.spot_mode = "live" if mode == "live" else "paper"
    ctx.spot = LiveSpotBroker(ctx.client) if mode == "live" else PaperSpotBroker(ctx.client, settings)


async def _restart_engine() -> None:
    proc = await asyncio.create_subprocess_exec("systemctl", "restart", ENGINE_SERVICE,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    if proc.returncode:
        raise HTTPException(500, f"restart failed: {err.decode(errors='ignore')[:300]}")


# The main engine's strategy profile: deploy/strategy.env copied here makes wallex-engine run the USDT-core strategy
# (the unit loads it after .env and mode.env); no file = the defaults (TMN markets).
STRATEGY_PROFILE_FILE = ROOT / "data" / "engine" / "profile.env"
STRATEGY_ENV = ROOT / "deploy" / "strategy.env"


def strategy_is_live() -> bool:
    return requested_engine_mode() == "live" and STRATEGY_PROFILE_FILE.exists()


class StrategyLiveIn(BaseModel):
    on: bool
    confirm: str = ""   # turning it on trades real money: must be "LIVE"


@app.post("/api/strategy/live")
async def set_strategy_live(body: StrategyLiveIn):
    """Run the strategy for real: the main engine switches to live with the strategy's settings (it then converts
    free TMN to USDT and trades the satellite on the real account). Off: the main engine drops the profile and keeps
    its current mode. The shadow strategy session keeps running either way."""
    if body.on and body.confirm != "LIVE":
        raise HTTPException(400, "running the strategy for real needs confirm=LIVE")
    _need_systemd()
    await _turn_strategy_live(body.on)
    return {"ok": True, "strategy_live": strategy_is_live()}


async def _turn_strategy_live(on: bool) -> None:
    STRATEGY_SCHEDULE_FILE.unlink(missing_ok=True)   # a manual switch replaces any scheduled one
    if on:
        STRATEGY_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STRATEGY_PROFILE_FILE.write_text(STRATEGY_ENV.read_text())
        _write_mode("live")
    else:
        STRATEGY_PROFILE_FILE.unlink(missing_ok=True)
    await _restart_engine()


# A scheduled switch to real: {"at": unix seconds}. The web app's ticker fires it once, then deletes it.
STRATEGY_SCHEDULE_FILE = ROOT / "data" / "engine" / "strategy-live-at.json"


def strategy_live_at() -> float | None:
    try:
        return float(json.loads(STRATEGY_SCHEDULE_FILE.read_text())["at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


async def _run_strategy_schedule() -> None:
    at = strategy_live_at()
    if at is None or time.time() < at:
        return
    STRATEGY_SCHEDULE_FILE.unlink(missing_ok=True)   # one shot, even if the restart fails
    if not strategy_is_live():
        await _turn_strategy_live(True)


class StrategyScheduleIn(BaseModel):
    at: str | None = None   # ISO-8601 with offset; None cancels
    confirm: str = ""


@app.post("/api/strategy/schedule")
async def schedule_strategy_live(body: StrategyScheduleIn):
    """Schedule the switch to running the strategy for real (same effect as the switch, at a set time)."""
    if body.at is None:
        STRATEGY_SCHEDULE_FILE.unlink(missing_ok=True)
        return {"ok": True, "live_at": None}
    if body.confirm != "LIVE":
        raise HTTPException(400, "scheduling real trading needs confirm=LIVE")
    try:
        at = datetime.fromisoformat(body.at.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, "at must be ISO-8601, e.g. 2026-10-23T12:00:00+03:30")
    if at.tzinfo is None or at.timestamp() <= time.time():
        raise HTTPException(400, "at must be in the future and carry a UTC offset")
    _need_systemd()
    STRATEGY_SCHEDULE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STRATEGY_SCHEDULE_FILE.write_text(json.dumps({"at": at.timestamp(), "iso": at.isoformat()}))
    return {"ok": True, "live_at": at.timestamp()}


_key_check: dict = {"at": 0.0, "main": None}


async def check_trading_key() -> bool:
    """Is a main-account key configured for a trade-credit sub-account? (orders would fail with 1006)"""
    if not (settings.wallex_api_key and settings.wallex_sub_account_client_id):
        return False
    if time.time() - _key_check["at"] > 600 or _key_check["main"] is None:
        _key_check.update(at=time.time(), main=await ctx.client.key_is_main_account())
    ctx.key_is_main = bool(_key_check["main"])
    return ctx.key_is_main


@app.get("/api/account")
async def account():
    """The real Wallex account (trade-credit sub-account when configured), read-only, plus the engine."""
    out: dict = {"has_key": bool(settings.wallex_api_key), "sub_account": bool(settings.wallex_sub_account_client_id)}
    if await check_trading_key():
        out["key_warning"] = ("کلید API فعلی مال حساب اصلی است؛ موجودی اعتبار معاملاتی را می‌خواند ولی سفارش‌ها با موجودی "
                              "حساب اصلی سنجیده و رد می‌شوند. برای حساب «هکاتون» یک کلید API با دسترسی معامله بسازید.")
    if settings.wallex_api_key:
        try:
            out["spot"] = await agent_tools.value_balances(ctx, await LiveSpotBroker(ctx.client).balances())
        except (WallexError, BrokerError) as e:
            out["error"] = str(e)
    try:
        out["engine"] = engine_status()
    except (OSError, ValueError, KeyError) as e:
        out["engine_error"] = str(e)
    eng = out.get("engine")
    if eng and eng["positions"]:
        with contextlib.suppress(WallexError):
            markets = await ctx.client.markets()
            usdt_tmn = float((markets.get("USDTTMN") or {}).get("stats", {}).get("lastPrice") or 0)
            for p in eng["positions"]:   # mark open positions to the last price
                last = float((markets.get(p["symbol"]) or {}).get("stats", {}).get("lastPrice") or 0)
                rate = usdt_tmn if p.get("quote") == "USDT" else 1.0
                if last:
                    p["mark_price"] = last
                    p["unrealized_pnl_tmn"] = round((last - p["entry_price"]) * p["qty"] * rate)
                    p["unrealized_pnl_pct"] = round(100 * (last / p["entry_price"] - 1), 2)
    return out


def _last_start(d: Path) -> dict:
    """The newest `start` event of a run: its config snapshot is the strategy actually running."""
    if not (d / "decisions.jsonl").exists():
        return {}
    with (d / "decisions.jsonl").open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        for back in (300_000, 3_000_000, size):
            f.seek(max(0, size - back))
            for line in reversed(f.read().decode(errors="ignore").splitlines()):
                if '"event": "start"' in line:
                    with contextlib.suppress(ValueError):
                        return json.loads(line)
            if back >= size:
                break
    return {}


_stretch_cache: dict = {}


def _stretch_start(d: Path, quote: str) -> float:
    """When the run's current setup began: the first start of the newest unbroken series of starts with this quote
    (restarts and deploys keep the curve; a switch between TMN and USDT markets begins a new one)."""
    f = d / "decisions.jsonl"
    key = (str(f), f.stat().st_size if f.exists() else 0, quote)
    if key in _stretch_cache:
        return _stretch_cache[key]
    begin = 0.0
    with contextlib.suppress(OSError):
        with f.open() as fh:
            for line in fh:
                if '"event": "start"' not in line:
                    continue
                with contextlib.suppress(ValueError):
                    e = json.loads(line)
                    q = (e.get("config") or {}).get("engine_quote", "TMN")
                    if q != quote:
                        begin = 0.0
                    elif not begin:
                        begin = e["ts"]
    _stretch_cache.clear()
    _stretch_cache[key] = begin
    return begin


@app.get("/api/strategy")
async def strategy():
    """The running strategy for the strategy panel: phase, USDT core, satellite limits, the end-of-challenge
    schedule, and equity next to two benchmarks (hold TMN = capital, hold USDT from the first mark)."""
    runs = [m for m in (requested_engine_mode(), STRATEGY_RUN, "shadow", "live") if engine_dir(m)]
    # the run with the USDT core is the strategy; prefer the app's own mode when both have it
    mode = next((m for m in runs if (engine_status(m) or {}).get("core")), runs[0] if runs else None)
    if not mode:
        return {"engine": None, "strategy_live": strategy_is_live(), "live_at": strategy_live_at()}
    eng = engine_status(mode)
    start = _last_start(engine_dir(mode))
    cfg = start.get("config") or {}
    rows = []
    with contextlib.suppress(OSError):
        rows = [r.split(",") for r in (engine_dir(mode) / "equity.csv").read_text().strip().splitlines()[1:]]
    if start.get("ts") and mode != STRATEGY_RUN:   # live/shadow dirs saw other setups: from when this quote began
        since = _stretch_start(engine_dir(mode), cfg.get("engine_quote", "TMN"))
        rows = [r for r in rows if float(r[0]) >= since - 60] or rows[-1:]
    step = max(1, len(rows) // 240)
    pts = [(int(r[0]), float(r[1])) for r in rows[::step]]
    if rows and (len(rows) - 1) % step:   # always end on the newest mark
        pts.append((int(rows[-1][0]), float(rows[-1][1])))
    usdt = []
    if pts:
        with contextlib.suppress(Exception):   # the benchmark is optional; the panel works without it
            candles = await ctx.client.candles("USDTTMN", "60", pts[0][0] - 3600, int(time.time()))
            usdt = [(c["t"], c["c"]) for c in candles]
    bench, j = [], 0
    for t, _ in pts:
        while j + 1 < len(usdt) and usdt[j + 1][0] <= t:
            j += 1
        bench.append(usdt[j][1] if usdt else None)
    base = next((b for b in bench if b), None)
    capital = eng["capital_tmn"]
    first = pts[0][1] if pts else capital
    # what the engine manages vs coins bought outside it (by hand / the chat): the latter is not the satellite
    managed = 0.0
    if eng["positions"]:
        with contextlib.suppress(WallexError, KeyError, ValueError, TypeError):
            markets = await ctx.client.markets()
            usdt_tmn = float(markets["USDTTMN"]["stats"]["lastPrice"])
            for p in eng["positions"]:
                last = float(markets[p["symbol"]]["stats"]["lastPrice"])
                managed += last * p["qty"] * (usdt_tmn if p.get("quote") == "USDT" else 1.0)
    follow = (eng.get("follow") or {}).get("holdings") or {}
    if follow:   # a copied trader's coins are the engine's, not "outside the engine"
        with contextlib.suppress(WallexError, KeyError, ValueError, TypeError):
            markets = await ctx.client.markets()
            usdt_tmn = float(markets["USDTTMN"]["stats"]["lastPrice"])
            for sym, qty in follow.items():
                managed += float(markets[sym]["stats"]["lastPrice"]) * qty * usdt_tmn
    pending = sum(float(o.get("amount_tmn") or 0) for o in eng["pending"])
    exposure = eng.get("exposure_tmn") or 0.0
    return {
        "return_pct": round(100 * (eng["equity_tmn"] / capital - 1), 3) if eng.get("equity_tmn") else None,
        "allocation": {"satellite_tmn": round(managed + pending),
                       "other_coins_tmn": round(max(0.0, exposure - managed - pending))},
        "mode": start.get("mode") or mode, "run": mode, "engine": eng, "strategy_live": strategy_is_live(),
        "paused": engine_paused(),
        "live_at": strategy_live_at(), "quote": cfg.get("engine_quote", "TMN"), "started": start.get("time"),
        "end_time": cfg.get("engine_end_time") or settings.engine_end_time or None,
        "params": {k.removeprefix("engine_"): cfg.get(k) for k in (
            "engine_max_exposure_pct", "engine_max_position_pct", "engine_max_positions", "engine_risk_per_trade_pct",
            "engine_breadth_gate_pct", "engine_drawdown_pause_pct", "engine_drawdown_stop_pct", "engine_core_guard_pct",
            "engine_core_exit_hours", "engine_core_exit_slices", "engine_flatten_before_end_min",
            "engine_no_entry_before_end_min", "engine_fee_rate", "engine_setups", "engine_decider") if k in cfg},
        "curve": [{"t": t, "equity": round(e), "hold_usdt": round(first * b / base) if b and base else None,
                   "hold_tmn": round(first)} for (t, e), b in zip(pts, bench)],
    }


# ---------------------------------------------------------- tuning, personas, strategy chat
TUNING_FILE = ROOT / "data" / "engine" / "tuning.env"   # loaded by wallex-engine.service after the profile
COPY_FILE = ROOT / "data" / "engine" / "copy.json"      # which persona the main engine copies: {"id", "at", "modified"}


def copying() -> dict | None:
    try:
        return json.loads(COPY_FILE.read_text())
    except (OSError, ValueError):
        return None


def effective_settings() -> dict:
    """What the main engine runs with: its last start's config (the restart applies saved changes), else this
    process's settings plus the saved overrides."""
    d = engine_dir(requested_engine_mode())
    cfg = (_last_start(d).get("config") or {}) if d else {}
    over = tuning.read_env(TUNING_FILE)
    out = {}
    for p in tuning.PARAMS:
        k = p["key"]
        v = cfg.get(k, over.get(k, getattr(settings, k, None)))
        try:
            out[k] = tuning.validate({k: v})[k]
        except tuning.TuningError:
            out[k] = v
    return out


def _engine_brief() -> dict:
    eng = engine_status(requested_engine_mode()) or {}
    return {k: eng.get(k) for k in ("mode", "equity_tmn", "cash_tmn", "exposure_tmn", "drawdown_pct", "regime",
                                    "regime_reason", "stats", "core")} | {
        "open_positions": [p.get("symbol") for p in eng.get("positions", [])],
        "recent_events": [{k: e.get(k) for k in ("time", "event", "symbol", "reason")} for e in eng.get("events", [])[:8]]}


@app.get("/api/strategy/tuning")
async def get_tuning():
    return {"params": tuning.PARAMS, "groups": tuning.GROUPS, "values": effective_settings(),
            "overrides": tuning.read_env(TUNING_FILE), "copying": copying(), "live": requested_engine_mode() == "live",
            "pending_restart": TUNING_FILE.exists() and TUNING_FILE.stat().st_mtime > (
                (engine_status(requested_engine_mode()) or {}).get("started_ts") or 0)}


class TuningIn(BaseModel):
    values: dict = {}
    reset: bool = False      # drop every saved override first (back to the strategy profile)
    confirm: str = ""        # "LIVE" when the engine trades real money


def _check_live_confirm(confirm: str) -> None:
    if requested_engine_mode() == "live" and confirm != "LIVE":
        raise HTTPException(400, "the engine trades real money: changing its settings needs confirm=LIVE")


@app.post("/api/strategy/tuning")
async def set_tuning(body: TuningIn):
    """Save setting overrides for the main engine and restart it so they take effect."""
    try:
        new = tuning.validate(body.values)
        merged = new if body.reset else {**tuning.read_env(TUNING_FILE), **new}
        tuning.validate(merged, new=False)
        tuning.check_pair({**effective_settings(), **merged})   # e.g. a new pause above the profile's stop
    except tuning.TuningError as e:
        raise HTTPException(400, str(e))
    _check_live_confirm(body.confirm)
    _need_systemd()
    _write_tuning(merged)
    cp = copying()
    if cp and not body.reset:
        COPY_FILE.write_text(json.dumps({**cp, "modified": True}))
    elif body.reset:
        COPY_FILE.unlink(missing_ok=True)
    await _restart_engine()
    return {"ok": True, "overrides": merged}


def _write_tuning(values: dict) -> None:
    TUNING_FILE.parent.mkdir(parents=True, exist_ok=True)
    if values:
        TUNING_FILE.write_text("# saved from the UI (strategy tuning / copy-trading); wallex_agent/tuning.py\n"
                               + tuning.to_env(values))
    else:
        TUNING_FILE.unlink(missing_ok=True)


def _persona_record(p: dict) -> dict:
    """A persona's public profile plus its track record from its own shadow journal."""
    rec = {k: p[k] for k in ("id", "name", "handle", "tagline", "bio", "glyph", "hue", "risk", "style", "preset")}
    rec["run"] = persona_mod.run_name(p)
    rec.update(_track_record(ROOT / "data" / "engine" / rec["run"]))
    bt = ROOT / "results" / "personas" / f"{p['id']}.json"
    with contextlib.suppress(OSError, ValueError):
        rec["backtest"] = json.loads(bt.read_text())
    return rec


def _open_symbols(state: dict) -> list:
    """Symbols held by a session. An engine session keeps its positions as a dict; a followed trader's copy
    (traders.Follower) keeps the trader's positions as a list, or None before its first poll, and holds its own coins
    in `holdings`."""
    pos = state.get("positions")
    if isinstance(pos, dict):
        return [x.get("symbol") for x in pos.values() if isinstance(x, dict)]
    holdings = state.get("holdings")
    return list(holdings) if isinstance(holdings, dict) else []


def _track_record(d: Path) -> dict:
    """A shadow session's results from its journal directory: equity curve, return, drawdown, trades."""
    rec: dict = {}
    rows = []
    with contextlib.suppress(OSError):
        rows = [r.split(",") for r in (d / "equity.csv").read_text().strip().splitlines()[1:]]
    trades = []
    with contextlib.suppress(OSError):
        with (d / "trades.csv").open() as f:
            trades = list(csv.DictReader(f))
    capital = settings.engine_capital_tmn
    eq = [float(r[1]) for r in rows]
    peak, dd = capital, 0.0
    for v in eq:
        peak = max(peak, v)
        dd = max(dd, 1 - v / peak)
    step = max(1, len(rows) // 80)
    pnls = [float(t["pnl_tmn"] or 0) for t in trades]
    state = {}
    with contextlib.suppress(OSError, ValueError):
        state = json.loads((d / "state.json").read_text())
    updated = (d / "state.json").stat().st_mtime if (d / "state.json").exists() else None
    rec.update({
        "active": bool(updated and time.time() - updated < 600),
        "since": int(rows[0][0]) if rows else None,
        "equity_tmn": eq[-1] if eq else None,
        "return_pct": round(100 * (eq[-1] / capital - 1), 2) if eq else None,
        "max_drawdown_pct": round(100 * dd, 2) if eq else None,
        "trades": len(trades), "wins": sum(1 for x in pnls if x > 0),
        "win_rate_pct": round(100 * sum(1 for x in pnls if x > 0) / len(pnls), 1) if pnls else None,
        "realized_pnl_tmn": round(sum(pnls)),
        "open_positions": _open_symbols(state),
        "core": (state.get("core") or {}).get("status"),
        "curve": [[int(r[0]), round(float(r[1]))] for r in rows[::step]] + ([[int(rows[-1][0]), round(float(rows[-1][1]))]]
                                                                             if rows and (len(rows) - 1) % step else []),
        "recent_trades": [{k: t.get(k) for k in ("symbol", "setup", "exit_time", "pnl_pct", "exit_reason")}
                          for t in reversed(trades[-6:])],
    })
    return rec


def _trader_record(t: dict) -> dict:
    """A followed real trader: their public Hyperliquid profile (trader.json, written by the follower) and our
    simulated copy's track record."""
    run = traders_mod.run_name(t["address"])
    d = ROOT / "data" / "engine" / run
    prof = {}
    with contextlib.suppress(OSError, ValueError):
        prof = json.loads((d / "trader.json").read_text())
    addr = t["address"]
    events = []
    with contextlib.suppress(OSError):
        with (d / "decisions.jsonl").open("rb") as f:   # the tail: what the trader did and what the copy did
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 60_000))
            for line in reversed(f.read().decode(errors="ignore").splitlines()[1:]):
                with contextlib.suppress(ValueError):
                    e = json.loads(line)
                    if e.get("event") in ("trader", "entry", "exit", "error"):
                        events.append({k: e.get(k) for k in ("time", "event", "symbol", "coin", "action", "side", "usd",
                                                              "copied", "reason", "error", "pnl_pct")})
                if len(events) >= 15:
                    break
    return {"id": run, "address": addr, "name": t.get("name") or prof.get("name"), "source": t.get("source"),
            "events": events,
            "hue": int(addr[2:8], 16) % 360, "run": run, **_track_record(d),
            **{k: prof.get(k) for k in ("account_usd", "positions", "fills", "board", "copy_weights", "not_copied",
                                         "updated")}}


@app.get("/api/traders")
async def list_traders():
    return {"traders": [_trader_record(t) for t in traders_mod.followed(traders_mod.DIR)]}


class TraderIn(BaseModel):
    address: str
    name: str = ""


@app.post("/api/traders")
async def add_trader(body: TraderIn):
    """Follow a trader by their Hyperliquid address (e.g. a well-known trader's public address); the follower service
    picks it up within a minute."""
    addr = body.address.strip().lower()
    if not traders_mod.ADDRESS.match(addr):
        raise HTTPException(400, "not a Hyperliquid (0x...) address")
    pinned = traders_mod.load_json(traders_mod.DIR / "pinned.json", [])
    pinned = [p for p in pinned if p.get("address", "").lower() != addr]
    if len(pinned) >= 8:
        raise HTTPException(400, "at most 8 added traders")
    pinned.insert(0, {"address": addr, "name": body.name.strip()[:40] or None, "added": time.time()})
    traders_mod.save_json(traders_mod.DIR / "pinned.json", pinned)
    return {"ok": True}


class TraderCopyIn(BaseModel):
    confirm: str = ""
    pct: float = 0.5


@app.post("/api/traders/{address}/copy")
async def copy_trader(address: str, body: TraderCopyIn):
    """Copy a real trader with the main engine: it follows their long positions with up to `pct` of equity (real money
    in live mode, so live needs confirm=LIVE). Restarts the engine; the challenge window settings are kept."""
    addr = address.lower()
    t = next((x for x in traders_mod.followed(traders_mod.DIR) if x["address"] == addr), None)
    if not t:
        raise HTTPException(404, "that trader is not followed; add it first")
    try:
        values = {**tuning.keep_on_copy(tuning.read_env(TUNING_FILE)),
                  **tuning.validate({"engine_follow_address": addr, "engine_follow_pct": body.pct})}
    except tuning.TuningError as e:
        raise HTTPException(400, str(e))
    _check_live_confirm(body.confirm)
    _need_systemd()
    STRATEGY_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STRATEGY_PROFILE_FILE.write_text(STRATEGY_ENV.read_text())   # the copy trades USDT markets with the USDT core
    _write_tuning(values)
    name = t.get("name") or f"نهنگ {addr[-4:].upper()}"
    COPY_FILE.write_text(json.dumps({"id": traders_mod.run_name(addr), "kind": "trader", "address": addr, "name": name,
                                     "pct": body.pct, "at": time.time(), "modified": False}, ensure_ascii=False))
    await _restart_engine()
    return {"ok": True, "copying": copying()}


@app.delete("/api/traders/{address}")
async def remove_trader(address: str):
    pinned = traders_mod.load_json(traders_mod.DIR / "pinned.json", [])
    traders_mod.save_json(traders_mod.DIR / "pinned.json", [p for p in pinned if p.get("address", "").lower() != address.lower()])
    return {"ok": True}


@app.get("/api/personas")
async def list_personas():
    return {"personas": [_persona_record(p) for p in persona_mod.PERSONAS], "copying": copying(),
            "live": requested_engine_mode() == "live", "strategy_live": strategy_is_live(),
            "traders": [_trader_record(t) for t in traders_mod.followed(traders_mod.DIR)]}


class CopyIn(BaseModel):
    confirm: str = ""


@app.post("/api/personas/{persona_id}/copy")
async def copy_persona(persona_id: str, body: CopyIn):
    """Copy a persona: the main engine takes its preset (on top of the strategy profile) and restarts. The app's
    mode is not changed: in live mode the engine then trades the persona's rules with real money."""
    p = persona_mod.BY_ID.get(persona_id)
    if not p:
        raise HTTPException(404, "unknown persona")
    _check_live_confirm(body.confirm)
    _need_systemd()
    STRATEGY_PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STRATEGY_PROFILE_FILE.write_text(STRATEGY_ENV.read_text())   # personas are built on the strategy profile
    _write_tuning({**tuning.keep_on_copy(tuning.read_env(TUNING_FILE)), **tuning.validate(p["preset"])})
    COPY_FILE.write_text(json.dumps({"id": p["id"], "name": p["name"], "at": time.time(), "modified": False},
                                    ensure_ascii=False))
    await _restart_engine()
    return {"ok": True, "copying": copying()}


@app.post("/api/personas/stop-copy")
async def stop_copy(body: CopyIn):
    _check_live_confirm(body.confirm)
    _need_systemd()
    COPY_FILE.unlink(missing_ok=True)
    _write_tuning(tuning.keep_on_copy(tuning.read_env(TUNING_FILE)))
    await _restart_engine()
    return {"ok": True}


# Strategy-assistant conversations, one JSON file each: {"id", "title", "updated", "messages": [...]}.
STRATEGY_CHATS = ROOT / "data" / "strategy_chats"


def _chat_path(cid: str) -> Path:
    if not cid.replace("-", "").isalnum():
        raise HTTPException(400, "bad conversation id")
    return STRATEGY_CHATS / f"{cid}.json"


def _load_chat(cid: str) -> dict:
    try:
        return json.loads(_chat_path(cid).read_text())
    except OSError:
        raise HTTPException(404, "no such conversation")


def _save_chat(chat: dict) -> None:
    STRATEGY_CHATS.mkdir(parents=True, exist_ok=True)
    chat["updated"] = time.time()
    _chat_path(chat["id"]).write_text(json.dumps(chat, ensure_ascii=False))


@app.get("/api/strategy/chats")
async def list_strategy_chats():
    rows = []
    for f in sorted(STRATEGY_CHATS.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:50] \
            if STRATEGY_CHATS.exists() else []:
        with contextlib.suppress(ValueError, OSError):
            c = json.loads(f.read_text())
            rows.append({"id": c["id"], "title": c.get("title", ""), "updated": c.get("updated"),
                         "turns": sum(1 for m in c.get("messages", []) if m.get("role") == "user")})
    return rows


@app.get("/api/strategy/chats/{cid}")
async def get_strategy_chat(cid: str):
    return _load_chat(cid)


@app.delete("/api/strategy/chats/{cid}")
async def delete_strategy_chat(cid: str):
    _chat_path(cid).unlink(missing_ok=True)
    return {"ok": True}


CONTROL_FILE = ROOT / "data" / "engine" / "control.json"   # the engine's on/off switch, read every bar and tick


def engine_paused() -> bool:
    try:
        return bool(json.loads(CONTROL_FILE.read_text()).get("paused"))
    except (OSError, ValueError):
        return False


class PauseIn(BaseModel):
    on: bool               # True = trading on, False = paused
    confirm: str = ""      # turning it back on in live mode trades real money: "LIVE"


@app.post("/api/engine/switch")
async def engine_switch(body: PauseIn):
    """Off pauses new entries and USDT buys (exits, stops and the final sell-off continue); no restart needed."""
    if body.on:
        _check_live_confirm(body.confirm)
    CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        ctl = json.loads(CONTROL_FILE.read_text())
    except (OSError, ValueError):
        ctl = {}
    CONTROL_FILE.write_text(json.dumps({**ctl, "paused": not body.on, "at": time.time()}))
    return {"ok": True, "on": not engine_paused()}


class ConsolidateIn(BaseModel):
    assets: list[str]
    confirm: str = ""      # selling real coins: "LIVE" in live mode


@app.post("/api/engine/consolidate")
async def engine_consolidate(body: ConsolidateIn):
    """Ask the engine to sell coins bought outside it into the strategy's quote (the USDT core) on its next tick."""
    _check_live_confirm(body.confirm)
    assets = sorted({a.strip().upper() for a in body.assets if a.strip().isalnum()} - {"TMN", "USDT"})
    if not assets:
        raise HTTPException(400, "no asset to consolidate")
    try:
        ctl = json.loads(CONTROL_FILE.read_text())
    except (OSError, ValueError):
        ctl = {}
    ctl["consolidate"] = sorted(set(ctl.get("consolidate") or []) | set(assets))
    CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONTROL_FILE.write_text(json.dumps(ctl))
    return {"ok": True, "pending": ctl["consolidate"]}


class StrategyChatIn(BaseModel):
    message: str
    session_id: str | None = None
    allow_apply: bool = True     # the assistant may set parameters itself when asked (undo is one click)
    confirm: str = ""            # "LIVE" when the engine trades real money


async def _apply_overrides(values: dict, confirm: str, source: str) -> dict:
    """Merge validated overrides into tuning.env and restart the engine. Returns the overrides before the change."""
    before = tuning.read_env(TUNING_FILE)
    merged = {**before, **tuning.validate(values)}
    tuning.validate(merged, new=False)
    tuning.check_pair({**effective_settings(), **merged})
    _check_live_confirm(confirm)
    _need_systemd()
    _write_tuning(merged)
    cp = copying()
    if cp:
        COPY_FILE.write_text(json.dumps({**cp, "modified": True}, ensure_ascii=False))
    await _restart_engine()
    return before


@app.post("/api/strategy/chat")
async def chat_strategy(body: StrategyChatIn):
    if not settings.llm_api_key:
        raise HTTPException(400, "LLM_API_KEY is not set")
    chat = _load_chat(body.session_id) if body.session_id else {
        "id": f"{int(time.time())}-{os.urandom(3).hex()}", "title": body.message.strip()[:60], "messages": []}
    chat["messages"].append({"role": "user", "content": body.message.strip(), "time": time.time()})
    cp = copying()
    effective = effective_settings()
    try:
        r = await strategy_chat.ask(settings, chat["messages"], effective, _engine_brief(), cp.get("name") if cp else None,
                                    allow_apply=body.allow_apply)
    except Exception as e:   # provider errors (budget, timeout) are shown in the chat, not as a 500
        raise HTTPException(502, f"LLM: {type(e).__name__}: {str(e)[:300]}")
    msg = {"role": "assistant", "content": r["reply"], "proposal": r["proposal"], "time": time.time()}
    if r["apply"]:
        try:
            msg["applied"] = {"changes": r["proposal"]["changes"],
                              "before": {k: effective.get(k) for k in r["proposal"]["changes"]},
                              "previous_overrides": await _apply_overrides(r["proposal"]["changes"], body.confirm, "assistant")}
        except HTTPException as e:
            msg["apply_error"] = e.detail
        except tuning.TuningError as e:
            msg["apply_error"] = str(e)
    chat["messages"].append(msg)
    _save_chat(chat)
    return {"session_id": chat["id"], "message": msg}


class UndoIn(BaseModel):
    index: int
    confirm: str = ""


@app.post("/api/strategy/chats/{cid}/undo")
async def undo_strategy_change(cid: str, body: UndoIn):
    """Undo what the assistant applied in message `index`: each setting it changed goes back to its saved value
    from before (or back to the profile's if it had none); settings changed since by anything else are kept."""
    chat = _load_chat(cid)
    try:
        msg = chat["messages"][body.index]
        prev = msg["applied"]["previous_overrides"]
        changed = list(msg["applied"]["changes"])
    except (IndexError, KeyError, TypeError):
        raise HTTPException(400, "that message did not apply anything")
    if msg.get("undone"):
        raise HTTPException(400, "already undone")
    _check_live_confirm(body.confirm)
    _need_systemd()
    now = tuning.read_env(TUNING_FILE)
    for k in changed:
        if k in prev:
            now[k] = prev[k]
        else:
            now.pop(k, None)
    try:
        tuning.validate(now, new=False)
    except tuning.TuningError as e:
        raise HTTPException(400, f"cannot undo: {e}")
    _write_tuning(now)
    await _restart_engine()
    msg["undone"] = time.time()
    _save_chat(chat)
    return {"ok": True}


@app.get("/api/alerts")
async def alerts():
    return ctx.alerts.list()


@app.post("/api/alerts/{alert_id}/ack")
async def ack_alert(alert_id: str):
    ctx.alerts.ack(alert_id)
    return {"ok": True}


class TicketIn(BaseModel):
    session_id: str | None = None
    market: str
    symbol: str
    side: str
    amount: float
    order_type: str = "MARKET"
    price: float | None = None
    leverage: float = 1
    take_profit: float | None = None
    stop_loss: float | None = None


@app.post("/api/ui/ticket")
async def submit_ticket(t: TicketIn):
    """A generative-UI trade ticket was submitted: route it through the same risk guard +
    confirmation flow as an agent-initiated trade."""
    if t.market == "spot":
        name = "spot_place_order"
        args = {"symbol": t.symbol, "side": t.side, "type": t.order_type, "quote_amount": t.amount,
                "price": t.price, "reason": "ثبت‌شده از برگه سفارش"}
    else:
        name = "futures_open_position"
        args = {"symbol": t.symbol, "side": t.side, "leverage": t.leverage, "margin": t.amount,
                "order_type": t.order_type, "price": t.price, "take_profit": t.take_profit,
                "stop_loss": t.stop_loss, "reason": "ثبت‌شده از برگه سفارش"}
    ok, result, confirm = await agent.run_tool(name, json.dumps(args), agent.session(t.session_id))
    return {"ok": ok, "result": result, "action": confirm}


@app.post("/api/paper/reset")
async def paper_reset():
    if isinstance(ctx.spot, PaperSpotBroker):
        ctx.spot.reset(settings.paper_usdt, settings.paper_tmn)
    if isinstance(ctx.futures, PaperFuturesBroker):
        ctx.futures.reset(settings.paper_usdt, settings.paper_tmn)
    return {"ok": True}


# ----------------------------------------------------------------- static
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    async def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        f = (DIST / path).resolve()
        if not f.is_relative_to(DIST.resolve()):
            raise HTTPException(404, "not found")
        return FileResponse(f if path and f.is_file() else DIST / "index.html")


def main():
    import uvicorn
    uvicorn.run("wallex_agent.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
