"""Copy-trading bots (added 17 Nov 2026): the MQL5 signal copy rebuilt from the signal page's
balance/equity series, and the Sonic-chain wallet copy."""
import json

import pandas as pd
import pytest

from fakes import FamilySource
from quantdesk.bots.families import copytrade as ct
from quantdesk.bots.runner import load_config, run_cycle

T0 = pd.Timestamp("2026-10-12T17:00Z")


def _samples(rows):
    s = pd.DataFrame(rows, columns=["ts", "balance", "equity"])
    s.index = pd.to_datetime(s.pop("ts"), utc=True)
    return s


NOFLOW = pd.DataFrame(columns=["ts", "amount"])


# ------------------------------------------------------------------------------------ page parsing
def test_parse_signal_page():
    html = ("<script>var equityData=[1760288400,1000,1000,1760306400,1000,1050.5,1760324400,1040,1040];</script>"
            "<span>Trades:</span> <b>483</b> Subscribers: <b>0</b>"
            "<script>var acc={growth:-0.4081,balance:1040,equity:1040,drawdown:0,depositLoad:0.05,deposits:0,"
            "withdrawals:510,currency:\"USD\",digits:2,stops:window.buildBalanceData([386,389,1760310000,-510]),};"
            "</script>")
    d = ct.parse_mql5_signal(html)
    assert list(d["samples"]["equity"]) == [1000, 1050.5, 1040]
    assert d["samples"].index[0] == pd.Timestamp(1760288400, unit="s", tz="UTC")
    assert d["flows"].iloc[0]["amount"] == -510 and d["flows"].iloc[0]["ts"] == pd.Timestamp(1760310000, unit="s", tz="UTC")
    assert d["acc"]["currency"] == "USD" and d["acc"]["trades"] == 483 and d["acc"]["withdrawals"] == 510
    with pytest.raises(ValueError):
        ct.parse_mql5_signal("<html>login required</html>")


# ------------------------------------------------------------------------------------ copier maths
def test_copy_gains_load_times_provider_return():
    s = _samples([("2026-10-12T16:00Z", 1000, 1000), ("2026-10-12T20:00Z", 1000, 1100), ("2026-10-13T02:00Z", 1100, 1100)])
    assert ct.copier_nav(s, NOFLOW, T0, load=1.0).iloc[-1] == pytest.approx(1.10)
    assert ct.copier_nav(s, NOFLOW, T0, load=0.95).iloc[-1] == pytest.approx(1.095)


def test_moves_before_the_start_do_not_count():
    s = _samples([("2026-10-12T10:00Z", 1000, 1000), ("2026-10-12T16:00Z", 1000, 1500), ("2026-10-12T20:00Z", 1000, 1500)])
    nav = ct.copier_nav(s, NOFLOW, T0, load=1.0)
    assert nav.iloc[0] == 1.0 and nav.iloc[-1] == pytest.approx(1.0)      # synchronised at 1500 equity: no gain since


def test_withdrawals_are_not_losses():
    s = _samples([("2026-10-12T16:00Z", 2000, 2000), ("2026-10-13T00:00Z", 1000, 1000), ("2026-10-14T00:00Z", 1100, 1100)])
    fl = pd.DataFrame({"ts": [pd.Timestamp("2026-10-12T22:00Z")], "amount": [-1000.0]})
    nav = ct.copier_nav(s, fl, T0, load=1.0)
    assert nav.iloc[1] == pytest.approx(1.0)                 # the provider took out half: the copier loses nothing
    assert nav.iloc[-1] == pytest.approx(1.10)               # and then copies the smaller account at twice the ratio


def test_ratio_resets_when_a_trade_closes():
    # +10% closed, then +10% of the new balance: the copy compounds (1.1 x 1.1), as MT5 resizes copies
    s = _samples([("2026-10-12T16:00Z", 1000, 1000), ("2026-10-12T20:00Z", 1100, 1100), ("2026-10-13T00:00Z", 1210, 1210)])
    assert ct.copier_nav(s, NOFLOW, T0, load=1.0).iloc[-1] == pytest.approx(1.21)


def test_copier_stopped_out_at_zero_stays_there():
    s = _samples([("2026-10-12T16:00Z", 1000, 1000), ("2026-10-12T20:00Z", 1000, -50), ("2026-10-13T00:00Z", 1000, 2000)])
    nav = ct.copier_nav(s, NOFLOW, T0, load=1.0)
    assert list(nav.iloc[1:]) == [0.0, 0.0]


def test_nav_candles_use_only_values_published_by_the_close():
    nav = pd.Series({T0: 1.0, pd.Timestamp("2026-10-12T17:20Z"): 1.2, pd.Timestamp("2026-10-12T17:25Z"): 0.9,
                     pd.Timestamp("2026-10-12T17:45Z"): 1.1})
    c = ct.nav_candles(nav, T0, pd.Timestamp("2026-10-12T18:10Z"))
    assert list(c.index) == list(pd.date_range(T0, periods=4, freq="15min"))          # 18:00 candle not closed yet
    assert list(c["close"]) == [1.0, 0.9, 1.1, 1.1]          # 17:45 belongs to the 17:30 candle (closes at 17:45)
    assert c.loc["2026-10-12T17:15Z", "high"] == 1.2 and c.loc["2026-10-12T17:15Z", "low"] == 0.9
    assert c.loc["2026-10-12T17:15Z", "open"] == 1.0


# ------------------------------------------------------------------------------------ the bot
def _cfg(only):
    cfg = load_config()
    cfg["bots"] = {b: s for b, s in cfg["bots"].items() if b in only}
    cfg["initial_capital"] = 5000.0
    return cfg


def test_mql5_bot_copies_and_pays_monthly_subscription(tmp_path):
    from quantdesk.bots import report
    cfg = _cfg(["mql5_exness_sonic"])
    src = FamilySource()
    s = run_cycle(tmp_path, now=pd.Timestamp("2026-12-20T12:00Z"), source=src, cfg=cfg)
    assert not s["errors"], s["errors"]
    b = s["bots"]["mql5_exness_sonic"]
    assert b["costs_eur"]["subscription"] == pytest.approx(3 * 30 / 1.16, rel=1e-6)     # 12 Oct, 12 Nov, 12 Dec
    fills = pd.read_csv(tmp_path / "fills.csv")
    assert len(fills) == 1 and fills.iloc[0]["ts"] == "2026-10-12T17:15:00+00:00"
    assert fills.iloc[0]["value_eur"] == pytest.approx(5000, rel=0.0015)
    # the account's value follows the copier NAV rebuilt from the page
    d = ct.parse_mql5_signal(src.mql5_page("2347199"))
    nav = ct.copier_nav(d["samples"], d["flows"], T0, 0.95)
    last = nav[nav.index <= pd.Timestamp("2026-12-20T12:00Z")].iloc[-1]
    units = fills.iloc[0]["qty"]
    assert b["equity_eur"] == pytest.approx(units * last + (5000 - units) - 90 / 1.16, rel=1e-6)
    # the subscription is a fee on the dashboard: P/L before fees adds it back
    data = report.load(tmp_path)
    t = report.summary_table(data, cfg).set_index("bot")
    assert t.at["mql5_exness_sonic", "fees"] == pytest.approx(90 / 1.16, rel=1e-6)
    assert t.at["mql5_exness_sonic", "pnl_gross"] == pytest.approx(t.at["mql5_exness_sonic", "pnl_net"] + 90 / 1.16, abs=1e-3)


def test_mql5_cycles_give_the_same_result_as_one_pass(tmp_path):
    cfg = _cfg(["mql5_exness_sonic"])
    one = run_cycle(tmp_path / "a", now=pd.Timestamp("2026-11-25T00:00Z"), source=FamilySource(), cfg=cfg)
    for now in ("2026-10-20T00:00Z", "2026-11-12T16:50Z", "2026-11-12T17:20Z", "2026-11-25T00:00Z"):
        many = run_cycle(tmp_path / "b", now=pd.Timestamp(now), source=FamilySource(), cfg=cfg)
    a, b = one["bots"]["mql5_exness_sonic"], many["bots"]["mql5_exness_sonic"]
    assert a["equity_eur"] == pytest.approx(b["equity_eur"], rel=1e-9)
    assert a["costs_eur"] == b["costs_eur"]


def test_failed_page_fetch_uses_the_stored_samples(tmp_path):
    cfg = _cfg(["mql5_exness_sonic"])
    run_cycle(tmp_path, now=pd.Timestamp("2026-11-01T00:00Z"), source=FamilySource(), cfg=cfg)
    stored = json.loads((tmp_path / "copy" / "mql5_2347199.json").read_text())
    assert stored["samples"] and stored["flows"]

    class Down(FamilySource):
        def mql5_page(self, signal_id):
            raise ConnectionError("mql5.com unreachable")
    s = run_cycle(tmp_path, now=pd.Timestamp("2026-11-02T00:00Z"), source=Down(), cfg=cfg)
    assert not s["errors"] and "fetch failed" in s["data_sources"]["mql5:2347199"]
    s = run_cycle(tmp_path / "new", now=pd.Timestamp("2026-11-02T00:00Z"), source=Down(), cfg=cfg)
    assert s["errors"] and "mql5.com unreachable" in s["errors"][0]          # nothing stored: the bot reports it


# ------------------------------------------------------------------------------------ Sonic chain maths
from quantdesk.bots import sonic_chain as sc  # noqa: E402
from fakes import FakeSonicRpc  # noqa: E402


def test_pool_prices_from_sqrt_price():
    f = FakeSonicRpc()
    assert sc.usd_price("S", f.sqrt_of("S/USD", 0.42)) == pytest.approx(0.42, rel=1e-9)
    assert sc.usd_price("ETH", f.sqrt_of("USD/ETH", 1 / 3900.0)) == pytest.approx(3900.0, rel=1e-9)


def test_price_impact_matches_concentrated_liquidity_maths():
    f = FakeSonicRpc(liq=7e17)
    sq = f.sqrt_of("S/USD", 0.5)
    sp = sq / sc.Q96
    # buying S pays USDC.e (token1) in: avg/spot - 1 = dy / (L sqrtP)
    assert sc.price_impact("S", "buy", 1000.0, sq, 7e17) == pytest.approx(1000e6 / (7e17 * sp), rel=1e-9)
    # selling S puts S (token0) in: 1 - 1/(1 + dx sqrtP / L)
    dx = 1000.0 / 0.5 * 1e18
    assert sc.price_impact("S", "sell", 1000.0, sq, 7e17) == pytest.approx(1 - 1 / (1 + dx * sp / 7e17), rel=1e-9)
    small, big = (sc.price_impact("ETH", "buy", u, f.sqrt_of("USD/ETH", 1 / 4000.0), 1e15) for u in (100.0, 10000.0))
    assert big == pytest.approx(100 * small, rel=0.02) and small > 0


def test_trade_flows_net_the_pools_in_one_transaction():
    f = FakeSonicRpc()
    # USDC.e -> S -> WETH in one transaction: S in and out cancels
    usd_s = f.swap_log("S/USD", 10, -(2000 * 10 ** 18), 1000 * 10 ** 6)        # pool pays 2000 S, gets 1000 USDC.e
    s_eth = f.swap_log("S/ETH", 10, 2000 * 10 ** 18, -(25 * 10 ** 16))          # pool gets 2000 S, pays 0.25 WETH
    fl = sc.trade_flows({"logs": [usd_s, s_eth]}, "0xabc")
    assert fl == {"USD": pytest.approx(-1000.0), "ETH": pytest.approx(0.25)}


def test_store_syncs_in_pieces_like_in_one_go(tmp_path):
    w = ["0x" + "11" * 20]
    start = pd.Timestamp("2026-10-12T17:00Z")
    now = pd.Timestamp("2026-10-16T00:00Z")
    one = sc.SonicStore(tmp_path / "a", w, start, FakeSonicRpc()).sync(now, chunk=200_000)
    st = sc.SonicStore(tmp_path / "b", w, start, FakeSonicRpc())
    for t in ("2026-10-13T00:00Z", "2026-10-14T09:07Z", "2026-10-16T00:00Z"):
        many = st.sync(pd.Timestamp(t), chunk=7_777)
    for coin in ("S", "ETH"):
        a, b = one.candles[coin], many.candles[coin]
        common = a.index.intersection(b.index)
        assert len(common) > 900
        pd.testing.assert_frame_equal(a.loc[common], b.loc[common])
        assert (a.index.to_series().diff().dropna() == pd.Timedelta(minutes=5)).all()
        assert a.index[0] <= start - pd.Timedelta(hours=20)
        assert a.index[-1] + pd.Timedelta(minutes=5) <= one.synced_ts
    ka = [(t["tx"], t["flows"], t["before"]) for t in one.trades if pd.Timestamp(t["ts"]) < now]
    kb = [(t["tx"], t["flows"], t["before"]) for t in many.trades if pd.Timestamp(t["ts"]) < now]
    assert ka == kb and len(ka) >= 12
    t = one.trades[0]
    assert t["wallet"] == w[0] and set(t["before"]) == {"S", "USD", "ETH"}


def test_store_refuses_another_wallet_selection(tmp_path):
    start = pd.Timestamp("2026-10-12T17:00Z")
    sc.SonicStore(tmp_path, ["0x" + "11" * 20], start, FakeSonicRpc()).sync(pd.Timestamp("2026-10-13T00:00Z"))
    with pytest.raises(RuntimeError):
        sc.SonicStore(tmp_path, ["0x" + "22" * 20], start, FakeSonicRpc()).sync(pd.Timestamp("2026-10-13T00:00Z"))


def test_rpc_down_keeps_what_was_synced(tmp_path):
    w = ["0x" + "11" * 20]
    start = pd.Timestamp("2026-10-12T17:00Z")
    ok = sc.SonicStore(tmp_path, w, start, FakeSonicRpc()).sync(pd.Timestamp("2026-10-13T00:00Z"))

    class Down(FakeSonicRpc):
        def block_number(self):
            raise ConnectionError("rpc down")
    res = sc.SonicStore(tmp_path, w, start, Down()).sync(pd.Timestamp("2026-10-14T00:00Z"))
    assert res.error and "rpc down" in res.error and res.synced_ts == ok.synced_ts
    assert len(res.candles["S"]) == len(ok.candles["S"])


# ------------------------------------------------------------------------------------ Sonic copy bot
def _sonic_engine():
    from quantdesk.bots.engine import BotEngine
    from quantdesk.bots.runner import _sonic_market
    from quantdesk.config import load_venues
    from quantdesk.costs import CostModel
    from quantdesk.ledger import FxRates
    markets = {}
    for iid, coin in ct.SONIC_COINS.items():
        _sonic_market(markets, iid, coin)
    return BotEngine(markets, CostModel(load_venues()), FxRates(1.16))


def _flat(px, pool, liq=1e30, n=40):
    f = FakeSonicRpc()
    human = px if pool == "S/USD" else 1 / px
    idx = pd.date_range(T0, periods=n, freq="5min")
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0,
                         "sqrt": float(f.sqrt_of(pool, human)), "liq": liq}, index=idx)


def test_sonic_bot_copies_the_share_of_holdings_the_wallet_trades():
    from quantdesk.ledger import Ledger
    w = "0x" + "ab" * 20
    bot = ct.SonicCopyBot("sonic", {"params": {"wallet": w, "gas_usd": 0.01, "onramp_fee": 0.003}})
    trades = [
        {"tx": "0x1", "wallet": w, "block": 100, "idx": 0, "ts": "2026-10-12T16:00:00+00:00",     # before the start
         "flows": {"USD": -500.0, "S": 1000.0}, "before": {"USD": 1000.0, "S": 0.0, "ETH": 0.0}},
        {"tx": "0x2", "wallet": w, "block": 200, "idx": 0, "ts": "2026-10-12T17:12:00+00:00",
         "flows": {"USD": -400.0, "S": 800.0}, "before": {"USD": 1000.0, "S": 0.0, "ETH": 0.0}},
        {"tx": "0x3", "wallet": w, "block": 300, "idx": 0, "ts": "2026-10-12T18:02:00+00:00",
         "flows": {"S": -400.0, "USD": 200.0}, "before": {"USD": 600.0, "S": 800.0, "ETH": 0.0}},
        {"tx": "0x4", "wallet": "0x" + "cd" * 20, "block": 310, "idx": 0, "ts": "2026-10-12T18:20:00+00:00",
         "flows": {"USD": -100.0, "ETH": 0.025}, "before": {"USD": 100.0, "S": 0.0, "ETH": 0.0}},   # another wallet
    ]
    bot.prepare(trades)
    eng = _sonic_engine()
    led = Ledger("sonic", 5000.0)
    candles = {"S_SONIC": _flat(0.5, "S/USD"), "ETH_SONIC": _flat(4000.0, "USD/ETH")}
    eng.run(bot, led, candles, T0, T0 + pd.Timedelta(days=5))
    usd0 = 5000 * 1.16 * (1 - 0.003)
    buy, sell = eng.fills
    assert buy.side == "buy" and buy.ts == "2026-10-12T17:15:00+00:00"         # seen at 17:15, filled at that open
    assert buy.qty * 0.5 * (1 + 0.0015) == pytest.approx(0.4 * usd0, rel=1e-3)   # 40% of the USDC.e, fee on top
    assert sell.side == "sell" and sell.ts == "2026-10-12T18:05:00+00:00"
    assert sell.qty == pytest.approx(0.5 * buy.qty, rel=1e-9)                  # the wallet sold half of its S
    assert led.costs["gas"] == pytest.approx(2 * 0.01 / 1.16)
    assert len(eng.fills) == 2                                                 # the other wallet is not copied


def test_sonic_bot_pays_price_impact_from_pool_liquidity():
    from quantdesk.ledger import Ledger
    w = "0x" + "ab" * 20
    bot = ct.SonicCopyBot("sonic", {"params": {"wallet": w}})
    bot.prepare([{"tx": "0x2", "wallet": w, "block": 200, "idx": 0, "ts": "2026-10-12T17:12:00+00:00",
                  "flows": {"USD": -1000.0, "S": 2000.0}, "before": {"USD": 1000.0, "S": 0.0, "ETH": 0.0}}])
    eng = _sonic_engine()
    led = Ledger("sonic", 5000.0)
    s = _flat(0.5, "S/USD", liq=7e17)
    eng.run(bot, led, {"S_SONIC": s, "ETH_SONIC": _flat(4000.0, "USD/ETH")}, T0, T0 + pd.Timedelta(days=5))
    (buy,) = eng.fills
    usd = 5000 * 1.16 * 0.997 / 1.0015                # all of it, less the pool fee
    k = sc.price_impact("S", "buy", usd, float(s["sqrt"].iloc[0]), 7e17)
    assert 0.001 < k < 0.2
    assert buy.price == pytest.approx(0.5 * (1 + k), rel=1e-7)


def test_sonic_bots_run_in_cycles_like_one_pass(tmp_path):
    bots = [f"sonic_wallet_copy.L{i}" for i in (1, 2)]
    cfg = _cfg(bots + ["sonic_wallet_copy.L3"])
    one = run_cycle(tmp_path / "a", now=pd.Timestamp("2026-10-16T00:00Z"), source=FamilySource(), cfg=cfg)
    assert not one["errors"], one["errors"]
    for now in ("2026-10-12T18:00Z", "2026-10-13T09:07Z", "2026-10-14T23:59Z", "2026-10-16T00:00Z"):
        many = run_cycle(tmp_path / "b", now=pd.Timestamp(now), source=FamilySource(), cfg=cfg)
        assert not many["errors"], many["errors"]
    for b in bots:
        a, m = one["bots"][b], many["bots"][b]
        assert a["equity_eur"] == pytest.approx(m["equity_eur"], rel=1e-9)
        assert a["last_candle"] == m["last_candle"] == "2026-10-15T23:55:00+00:00"
    fills = pd.read_csv(tmp_path / "a" / "fills.csv")
    assert set(fills["bot"]) == set(cfg["bots"])                  # every wallet's trades were copied
    assert (pd.to_datetime(fills["ts"]) >= T0).all()
    assert "rpc.soniclabs.com" in one["data_sources"]["sonic"]
