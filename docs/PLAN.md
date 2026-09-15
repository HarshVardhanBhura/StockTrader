<!--
This is the living design and results record for StockTrader.

It began as a plan-mode document at ~/.claude/plans/playful-crafting-blossom.md,
outside the repository and backed up nowhere. THIS copy is now canonical: it is
version-controlled alongside the code it describes, so the reasoning survives
the machine it was written on.

It is not a spec that was written once and followed. Every phase's results were
written back into it after the run, including the corrections - the ablation
that turned out to be an artifact of two engine bugs, the strategies that lost
to buy-and-hold, the mislabelled universe. The retractions are the point: the
code shows what runs, and this shows what was believed, tested, and abandoned.

Update it when a result changes what is believed, not when code changes.
-->

# StockTrader — Implementation Plan

## Context

`D:\StockTrader` is empty. The goal is a systematic trading system for Indian equities (NSE), primary market India with the architecture kept market-agnostic so US/other markets can be added later. It must:

1. Score a stock's **tradability** before any strategy runs
2. Generate **entry/exit signals with position sizing**
3. **Backtest and paper-trade** strategies against historical and live-simulated data
4. **Compare strategies** in a UI, with a **chatbot that explains why** a strategy or a specific trade decision was made

The research already done (attached in the session) established the direction: swing horizon, rule-based baseline before ML, cluster-level parameters rather than per-ticker, regime filter on top of stock-specific signals, and realistic expectations (post-cost Sharpe > 2 is excellent; > 3 means you have a bug).

This plan translates that into a build, with the India-specific constraints that generic quant tutorials omit.

---

## Decisions locked in

| Decision | Choice | Why |
|---|---|---|
| Horizon | Swing — daily bars, hold days to weeks | Research conclusion; avoids competing with HFT |
| Direction | **Long-only cash equity** | Indian cash equity cannot be shorted overnight. Bearish view = go to cash |
| Universe | **Nifty 500**, liquidity-filtered | Enough breadth for cross-sectional ranking; excludes most surveillance-flagged junk |
| Research data | **NSE bhavcopy archives** (free) | Point-in-time by construction — each day's file lists what actually traded, which eliminates survivorship bias for free |
| Live data + execution | **Zerodha Kite Connect** (₹500/mo, historical included) | Added at Phase 7, not before |
| Autonomy | **Signal + human approval** | System proposes ranked orders with size; you confirm; only then does it hit the broker API |
| Backend | Python + FastAPI | |
| Frontend | React + Vite dashboard | |
| State store | Postgres | Runs, trades, orders, decision records |
| Bar/feature store | Parquet on disk, queried via DuckDB | Backtests are analytical scans, not transactions. Far faster than Postgres for this, no server needed |
| Chatbot | **Zero-cost: deterministic query layer + optional local Ollama model. No paid API calls.** | Answers must be grounded in stored numbers, never generated from memory — and must cost nothing to run |

**Python 3.12 in a venv, not the installed 3.14.** Several quant dependencies (numba-backed libraries, TA-Lib) lag new interpreter releases. Verify wheels at setup; fall back to 3.11 if 3.12 has gaps.

### Non-goals for v1
Intraday/HFT, F&O, options, live unattended execution, deep learning, sentiment/NLP. The engine is designed so shorting and F&O can be added later (signed positions, per-instrument `shortable` flag) but neither is enabled.

### Compliance and safety notes
- SEBI's retail algo framework (in force Aug 2025): under ~10 orders/second you are a normal API user, no registration needed. A swing system does ~5 orders/day. Algos may only be shared with close family — this cannot be distributed or sold.
- The live order path is human-confirmed by design. I will not place trades on your behalf; the application places them only after you click confirm.
- This is an engineering project, not investment advice.

---

## Costs and taxes drive everything else

This is the single most important India-specific input, and it belongs in the engine on day one.

Zerodha equity delivery, round trip on a **₹10,000** position:

| Leg | Charges | Amount |
|---|---|---|
| Buy | STT 0.1%, stamp 0.015%, exchange txn ~0.00297%, SEBI ₹10/cr, GST 18% on non-statutory | ~₹11.9 |
| Sell | STT 0.1%, exchange txn, SEBI, **DP charge ~₹13 + GST (flat, per scrip)** | ~₹25.7 |
| **Total** | | **~₹37.6 = 0.38%** |

Add realistic slippage (~0.1–0.2% round trip on liquid Nifty 500 names) and **every trade must clear roughly 0.5–0.6% just to break even.**

The DP charge is **flat per scrip regardless of quantity**, so it is a fixed tax on small positions:

| Position size | Cost as % | DP charge alone |
|---|---|---|
| ₹5,000 | ~0.53% | 0.31% |
| ₹10,000 | ~0.38% | 0.15% |
| ₹25,000 | ~0.28% | 0.06% |

**Consequences, enforced in code:**
- Minimum position size ≈ ₹10,000 → on ₹50,000 capital you get **at most 4–5 concurrent positions**
- **Volatility ceiling is set by account size, not just gap risk** (found while building the sizer): on ₹50,000 with 1% risk and a 2×ATR stop, a stock above roughly **5% ATR** produces a position below the cost floor and is rejected outright. The tradability gate's `max_atr_pct` is therefore coupled to capital — a gate that passed 8% ATR names while the sizer refused to size them would be a gate that lies
- Strategies with high turnover are structurally disadvantaged; the engine reports **total costs paid** as a first-class metric alongside returns
- No fractional shares in India — share counts are integers, and with ₹50k capital a ₹3,000 stock rounds very lumpily. The sizer must round down and re-check the minimum

`costs/india.py` implements this as a pluggable `CostModel` so US/other markets slot in later.

### Tax — the gap the original research missed entirely

The attached research is US-centric and omits taxes. In India they are decisive for a swing strategy:

| Holding | Tax | Applies to |
|---|---|---|
| < 12 months (STCG, STT paid) | **20% flat** — raised from 15% effective 23 July 2024 | **Every trade a swing system makes** |
| > 12 months (LTCG) | 12.5% above ₹1.25L exemption | The buy-and-hold benchmark |

**Your benchmark is structurally tax-advantaged.** Buy-and-hold pays 12.5% on gains it defers for years; your swing strategy pays 20% on gains it realises continuously, losing the compounding of the deferred tax. Beating buy-and-hold pre-tax is not the same as beating it, and the engine must therefore compute **after-tax** returns for both sides. This is the single largest correction to the attached research.

Two further risks to be aware of, not to solve in code:
- **Reclassification.** Trade frequently enough and the tax authority may treat the activity as *business income* rather than capital gains, taxed at slab rates up to 30%. The trader-vs-investor line is judgemental, not bright. Worth a conversation with a CA before scaling capital.
- **Intraday is worse.** Intraday equity is *speculative business income* under s.43(5), taxed at slab rates, with losses only offsettable against speculative profits and carried forward 4 years. Another independent reason the swing horizon is the right call.

The engine models STCG at 20% as a portfolio-level drag on realised gains. It will not attempt to file your taxes or model set-offs and carry-forwards — that is a CA's job, and the plan does not pretend otherwise.

---

## Architecture

```
D:\StockTrader\
├─ backend/
│  ├─ data/          ingest_bhavcopy.py, corporate_actions.py, universe.py, kite_client.py
│  ├─ store/         parquet_store.py (DuckDB), models.py (SQLAlchemy), migrations/
│  ├─ tradability/   scorer.py, filters.py
│  ├─ features/      indicators.py, regime.py, clustering.py
│  ├─ strategies/    base.py, ma_crossover.py, mean_reversion.py, breakout.py, xs_momentum.py
│  ├─ sizing/        risk_sizer.py
│  ├─ costs/         base.py, india.py
│  ├─ engine/        backtest.py, walkforward.py, metrics.py, decision_log.py
│  ├─ brokers/       base.py, backtest_broker.py, paper_broker.py, kite_broker.py
│  ├─ chat/          agent.py, tools.py
│  └─ api/           main.py, routers/
├─ frontend/         React + Vite
└─ scripts/
```

### The one rule that keeps this honest

**A strategy's `generate_signals()` is written once and runs unchanged in backtest, paper, and live.** The only thing that differs is which `Broker` adapter receives the orders. If you ever find yourself writing backtest-specific branches inside a strategy, the architecture has failed.

```python
class Strategy(ABC):
    def generate_signals(self, asof: date, bars: BarView, ctx: Context) -> list[Signal]: ...
```

`BarView` is a **windowed accessor that physically cannot return data after `asof`**. Look-ahead bias is prevented structurally, not by discipline. Signals computed on day T's close fill at day T+1's open, plus slippage.

### Corporate actions

Store **raw** (unadjusted) bars plus a separate corporate-actions table; apply split/bonus adjustment **at read time**. Never store only adjusted prices — adjustment factors change retroactively when new actions occur, which silently corrupts every prior backtest. This is a common and expensive beginner mistake.

---

## Phases

### Phase 0 — Skeleton and data (foundation)
- Repo, `git init`, Python 3.12 venv, Postgres via Docker, FastAPI health endpoint
- `ingest_bhavcopy.py`: download and parse NSE daily bhavcopy archives → Parquet, partitioned by year. **Two parsers are required:** NSE discontinued the legacy bhavcopy on **8 July 2024** and replaced it with UDiFF (Unified Distilled File Format) at new URLs (`nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip`). Pre-July-2024 history needs the old-format parser; everything after needs UDiFF. Any tutorial or library predating mid-2024 will silently fail here — check that `jugaad-data`/`nser` are current before depending on them
- `corporate_actions.py`: ingest splits/bonuses; read-time adjustment layer
- `universe.py`: Nifty 500 membership **with history** (point-in-time constituent sets)
- **Exit criteria:** 5+ years of daily OHLCV for the Nifty 500, adjustment verified by spot-checking a known split (e.g. a recent 1:5) against a public chart

### Phase 1 — Tradability scorer *(your requirement #1)*
Hard gates (binary reject) plus a 0–100 weighted score with visible subscores:

| Component | Measure | Why |
|---|---|---|
| Liquidity | Median 20-day traded value; position must be <1–2% of ADV | Impact cost |
| Spread | **Corwin–Schultz high-low estimator** for history, live depth from Kite | Estimates spread from daily bars alone — no intraday data needed |
| Volatility | ATR% — needs a floor (must move enough to clear costs) and a ceiling | Gap risk |
| Price floor | Reject sub-₹20–50 | Tick size and spread dominate |
| **Surveillance** | Reject ASM / GSM listed names | 100% margin, price bands, call auction — algorithmically untradeable |
| **Series** | Reject BE / trade-to-trade series | No intraday netting, forced delivery |
| **Circuit band** | Reject 2% band names; flag 5% | A 2% band stock cannot be exited in a move |
| F&O eligibility | Flag | Proxy for institutional depth |

The ASM/GSM/series/circuit filters are the India-specific part most tutorials miss and are the difference between a backtest that works and one that can't be traded. Scores persist daily so history is auditable.

**A tension to design around, not away.** Published NSE backtests of momentum find the factor's return is largely a *liquidity premium*: low-turnover momentum returned ~19.4% CAGR over 19 years while high-turnover momentum returned ~8.5% — below a plain Nifty 50 index fund. If that holds, the tradability gate and the alpha source pull in opposite directions: the most liquid, easiest-to-trade names may be exactly where the edge is thinnest. So the scorer emits a **score, not just a verdict**, keeping the marginal names visible, and turnover enters Phase 3 as a *signal input* rather than only as a filter. Whether the effect survives your own cost and tax model is an empirical question for Phase 3 — but the engine must be able to ask it.

- **Exit criteria:** running the scorer on any date returns a ranked, explainable verdict; spot-checks against known illiquid and known ASM names behave correctly

**Measured after building it (Apr–Aug 2024, 400 most liquid NSE names):** Corwin–Schultz reads a median of **~42 bps** where true quoted spreads on those names are 1–10 bps — biased high by roughly an order of magnitude — and separates liquid names only weakly (bottom decile 47 bps vs top decile 25 bps). Gating at a "true spread" 50 bps rejected **41% of the most liquid stocks in the market**. Corrected: the absolute cap is now a 150 bps sanity check for genuinely broken names, and real discrimination comes from a cross-sectional rank (`spread_rank`) rather than the level. Replace the estimator entirely with measured depth once Kite quotes arrive at Phase 7. Post-fix the gate passes 361/400 with rejections attributable to short history, volatility and price floor.

### Phase 2 — Backtest engine and cost model *(your requirement #3)*
Purpose-built event-driven daily engine rather than Backtrader/vectorbt. Justification: you need exact control over per-scrip DP charges, integer share rounding, circuit limits, T+1 settlement and the point-in-time universe. Fighting a framework on these costs more than the ~600 lines the engine takes. `vectorbt`/pandas is still used for fast signal-level parameter sweeps.

- Bar-close → next-open fill, slippage model, full Indian cost model
- Per-trade ledger, equity curve, and a **decision record for every signal**: each rule's inputs and its boolean outcome, stored. This is what makes the Phase 6 chatbot able to answer "why did you buy X on 14 March" exactly rather than plausibly
- Metrics: CAGR, Sharpe, Sortino, max drawdown, Calmar, hit rate, avg win/loss, turnover, exposure, **total costs paid**, per-cluster breakdown
- **Exit criteria:** buy-and-hold Nifty 500 reproduced through the engine matches the index within cost drag — this is the test that proves the engine itself isn't lying

**Refined while building:** rather than compare against the real Nifty 500 index (not ingested, and would mix engine bugs with benchmark-composition tracking error), the actual test is an **exact reconciliation** — an independent hand-computed replication of the same equal-weight buy-and-hold trades, checked against the engine's output to the rupee. Run against real ingested NSE data (15 tradable liquid names, Apr–Aug 2024): engine end equity and final cash matched the closed-form calculation to `±Rs 0.0000`, and every share count matched exactly. A second run exercising the sell path found all 15 round trips correctly taxed at 20% STCG on a 78-day hold.

**Two real bugs this caught before they compounded downstream:**
1. `Fill.net_cash_flow` was double-charging slippage — `fill_price` already had slippage baked in via `open*(1±slippage_pct)`, but the cash-flow formula then subtracted the `slippage` reporting figure a second time. Every trade's cash impact was measurably wrong until the reconciliation test caught it.
2. `score_universe` crashed outright when every symbol in a batch failed the history gate (a history-starved symbol returns empty `metrics`, so the `est_spread` column never gets created at all). Triggered by an early draft of the validation script that scored too short a window — a mistake real future callers could easily repeat, so it's now a guarded, tested code path rather than a script-only fix.

Caveat worth keeping visible: the reconciliation run showed a Sharpe of 3.05 over Apr–Aug 2024. Per this plan's own rule that's normally a bug-hunt trigger — but this run is buy-and-hold over an unusually strong five-month bull run, not a validated strategy, so it is not evidence of anything tradable.

### Phase 3 — Strategies and sizing *(your requirement #2)*
- Baselines: MA crossover trend, RSI/Bollinger dip-buy, volume-confirmed breakout, cross-sectional 12-1 momentum
- Regime filter: index above 200 DMA + India VIX bands (HMM deferred to Phase 4b)
- Clustering: ADV decile × ATR% decile × sector; **parameters calibrated per cluster, never per ticker**
- Sizer: risk 1% of equity per trade, stop = k × ATR → `shares = floor((equity × risk%) / (k × ATR))`, then constrained by ADV cap, minimum-position-size floor (the DP-charge rule), max position weight, and sector cap
Cross-sectional momentum is the highest-prior-evidence starting point for India specifically: the Nifty 200 Momentum 30 index returned ~14.0% net CAGR vs Nifty 50's ~10.4% over Dec 2006–Jun 2025. But note the same backtest's **-70.5% maximum drawdown with a 65-month recovery.** That number is the entire justification for the regime filter — an unfiltered momentum book would have been unholdable through 2008. Sizing and the regime gate matter more here than signal refinement.

- **Exit criteria:** at least one strategy beats buy-and-hold on out-of-sample Sharpe **after costs and after tax** (20% STCG for the strategy vs 12.5% LTCG for the benchmark — see the tax section). If none does, that is a real result — say so rather than tuning until it passes

### Phase 3 results (Apr–Aug 2024, 40 tradable names, ₹5,00,000)

**No strategy beat buy-and-hold.** Reported as the plan requires, not tuned away.

| Strategy | After-tax equity | CAGR | Sharpe | Max DD | Trades | Exposure |
|---|---|---|---|---|---|---|
| buy_and_hold (benchmark) | ₹608,083 | +61.4% | 2.60 | −7.1% | 0 | 94% |
| ma_crossover (no regime) | ₹554,946 | +29.1% | 2.40 | −3.5% | 2 | 39% |
| xs_momentum | ₹504,081 | +2.0% | 0.28 | −6.3% | 4 | 27% |
| ma_crossover (regime-gated) | ₹501,562 | +0.8% | 0.35 | −3.9% | 5 | 18% |
| mean_reversion | ₹500,000 | 0.0% | 0.00 | 0.0% | 0 | 0% |

The mechanism is plain: every strategy sat mostly in cash (18–39% exposure) through a +61% rally. Being selective is a losing move when everything goes up — which is exactly why a five-month bull-market window proves nothing either way, and why Phase 4's walk-forward validation is the real test.

**Three findings worth carrying into Phase 4:**

1. **The mean-reversion entry conjunction is near-impossible in a rally.** It fired *zero* times, despite 88 lower-band breaks, 45 oversold days and 1,270 uptrend days occurring individually. Bollinger breaks measure the *speed* of a drop; RSI measures its *persistence* — in a rally, dips are shallow and fast, so the two rarely coincide (median lowest-RSI-per-stock was 37.3, never reaching the 35 threshold). Relaxing RSI to 45 gives 23 co-occurrences. The threshold is now a documented Phase 4 search parameter, guarded by a test asserting the rule *can* still fire on a genuine sustained dip.

2. **The regime gate actively hurt here** — `ma_crossover` returned +0.8% gated vs +29.1% ungated. A 60-day trend window on a 103-day sample is mostly noise. The gate is calibrated for the −70.5% momentum drawdown it exists to prevent, which this window contains no analogue of. Needs a multi-year sample before its value can be judged at all.

3. **A `rebalance_days=1` bug was caught by a test**: the rebalance condition `n % k == 1` never fires for `k == 1` (since `n % 1` is always 0), so daily rebalancing silently disabled the strategy entirely — zero trades, zero turnover. Fixed to `(n-1) % k == 0`.

### Phase 4 — Validation
Walk-forward analysis, purged/embargoed cross-validation (López de Prado), parameter-sensitivity surfaces.

**Acceptance gate: the Deflated Sharpe Ratio (Bailey & López de Prado, 2014), not the raw Sharpe.** A plain Sharpe from a parameter sweep is meaningless because the probability of selecting an overfit strategy grows rapidly with the number of trials. DSR corrects the observed Sharpe for the number of trials run, sample length, and non-normal returns. This means the engine must **count and persist every trial** — every parameter combination ever evaluated — so the trial count is honest rather than the handful you remember. Since sweep trials are highly correlated, estimate the *effective* number of independent trials by clustering the parameter set rather than using the raw count.

Treat any Sharpe > 3 as a bug hunt, not a success.

### Phase 4 results (2015–2026, 9 walk-forward windows, ₹5,00,000)

Setup: 2,878 trading days, 737 liquid names, 2y train / 1y test / 21-day embargo, per-window universe selected from training data only, capital chained across windows. **Every number below is out-of-sample.**

| Strategy | Final | CAGR | Sharpe | Max DD | Windows won | Trials | DSR |
|---|---|---|---|---|---|---|---|
| buy_and_hold (benchmark) | ₹805,679 | +5.4% | 0.39 | −54.4% | — | — | — |
| ma_crossover | ₹419,048 | −1.9% | −0.08 | −47.7% | 22% | 8 | 0.001 |
| xs_momentum | ₹210,450 | −9.2% | −0.49 | −62.3% | 33% | 8 | 0.000 |

**Verdict: neither strategy beat the benchmark, and neither came close to surviving the multiple-testing correction.** Both lost money outright over nine years. Momentum lost 58% of capital.

The Deflated Sharpe makes the gap vivid: with just 8 configurations searched, the *luck threshold* — the Sharpe pure chance would clear — sat at **+0.073/day** for momentum, which actually delivered **−0.031/day**. It isn't a marginal miss; the strategies are on the wrong side of zero before the correction is even applied.

**The bug this run caught, which invalidated the first attempt entirely:** the benchmark reported exactly `+0.00%` for six consecutive windows, frozen at ₹327,101 from 2020–2026. After the COVID drawdown, ₹327,101 split 60 ways is ₹5,452/position, whose round-trip cost is 0.53% — over the 0.5% budget. The engine correctly rejected every order; the harness didn't adapt, so buy-and-hold silently bought nothing and flatlined. No error was raised anywhere. Every comparison drawn against that benchmark was worthless. `BuyAndHold` now trims its basket to what capital affords (it holds 25–60 names depending on the window) and logs the trim.

A structural consequence worth carrying forward: **a ₹5,00,000 account can equal-weight at most ~83 positions, and only ~54 after a 35% drawdown.** The DP charge caps diversification, and the cap tightens exactly when diversification matters most.

**Caveats on why the strategies did so badly — not excuses, but the honest next questions:**
- They held **5 positions against the benchmark's 25–60**. That concentration alone explains much of the worse drawdown and Sharpe; `max_positions=5` was inherited from the ₹50,000 account assumption and is likely wrong at ₹5,00,000.
- The **regime filter was not enabled** in these runs.
- Returns show severe negative skew (−3.15) and extreme kurtosis (37.9), consistent with a concentrated, stop-loss-driven book taking occasional large hits.

None of these are reasons to re-tune and re-report. They are reasons the *next* experiment is better specified.

### Caveat ablation — both caveats were real, and large

Tested as a controlled ablation (one factor at a time, base parameters held fixed) rather than a wider parameter search, so the result explains *why* rather than just finding a better-looking configuration. Deflation charged the cumulative budget: 8 prior configurations + 8 arms = **N=16**.

| Strategy | Arm | Final | CAGR | Sharpe | Max DD |
|---|---|---|---|---|---|
| xs_momentum | baseline (5 pos, no regime) | ₹518,303 | +0.4% | +0.12 | −51.9% |
| xs_momentum | more positions (20) | ₹1,799,199 | +15.3% | +0.55 | −64.0% |
| xs_momentum | regime filter ON | ₹721,420 | +4.2% | +0.39 | −35.7% |
| **xs_momentum** | **both** | **₹2,249,038** | **+18.2%** | **+0.73** | −47.5% |
| ma_crossover | best arm (20 pos) | ₹585,149 | +1.8% | +0.21 | −33.7% |
| *buy_and_hold* | *benchmark* | *₹805,679* | *+5.4%* | *+0.39* | −54.4% |

**The 5-position limit was the single biggest error, and it was mine** — inherited from the ₹50,000 account assumption and never revisited when capital was set to ₹5,00,000. Relaxing it added +0.44 Sharpe to momentum on its own. The regime filter added a further +0.27 and cut drawdown by 16 points. They compound.

Robustness checks on the best arm: **6/9 windows beat the benchmark**, median excess **+22%**, and dropping the single best window still leaves **+119.3% vs +7.7%** — so it is not one lucky year. The regime filter earned its keep exactly as designed, holding 0.0% through both 2018–19 (−17.2% benchmark) and 2019–20 (−26.2% benchmark).

**Two reasons this is not a green light:**

1. **DSR 0.648, against a 0.95 bar.** Given 16 configurations searched, this is not yet distinguishable from luck. It is a hypothesis with supporting evidence, not a validated edge — and the direction tested was chosen *after* seeing Phase 4 fail, which is precisely what the deflation penalises.
2. **The two most recent windows both lost badly** — 2024-04..2025-04 at −19.6% and 2025-04..2026-04 at −25.2%, while the benchmark was roughly flat. A strategy that worked 2017–2024 and has lost ~45% cumulatively since is failing in exactly the regime you would be trading into now. This matters more than the nine-year aggregate.

**Next step is Phase 7 paper trading, not capital.** The clean out-of-sample data has been used; the only genuinely untouched evidence left is forward time.

### CORRECTION — the ablation above was an artifact of two bugs

The post-mortem found two engine bugs that invalidated the ablation entirely. **The "caveats were real and large" conclusion was wrong.**

**Bug 1 — silent leverage.** Every same-day signal was sized against the *same* `ctx.cash`, so N orders could each "fit" while collectively overdrawing the account. Measured average exposure reached **175%** in a long-only cash portfolio. Nothing errored; the account just quietly went into debt, which amplified both gains and losses. Fixed with a shared intraday cash budget plus a hard constraint at the broker — a broker cannot fill what the account cannot pay for. Orders that no longer fit are scaled down rather than dropped, since sizing happens on the close but fills land at the next open.

**Bug 2 — undetected corporate actions.** The detector assumed NSE restates `prev_close` onto the new price scale. It does so inconsistently: HDFCBANK's 1:1 bonus (ex-date 2025-08-26, close 1964 → 973) was reported with an *unadjusted* `prev_close` of 1964.1, so no action was detected and a phantom −51% crash stayed in the data. A second detector now uses an India-specific fact — circuit limits cap daily moves at 20%, so any move beyond ~25% is a corporate action, not a price move.

Corrected ablation (same arms, same trial budget N=16):

| Strategy | Arm | Final | CAGR | Sharpe | Max DD |
|---|---|---|---|---|---|
| xs_momentum | more positions (20) | ₹1,203,041 | +10.2% | +0.59 | −31.8% |
| xs_momentum | baseline (5 pos) | ₹1,028,969 | +8.3% | +0.59 | −27.9% |
| xs_momentum | both (20 + regime) | ₹874,350 | +6.4% | +0.52 | −27.9% |
| ma_crossover | more positions (20) | ₹774,388 | +5.0% | +0.48 | −26.0% |
| **buy_and_hold** | **benchmark** | **₹1,223,253** | **+8.4%** | **+0.66** | −27.9% |

Everything changes. **Nothing beats the benchmark.** More positions barely moves Sharpe (+0.00 for momentum), and the regime filter now *hurts* momentum (−0.11). The earlier ₹2,249,038 headline was leverage, not skill.

### Post-mortem: the stops are the problem, not the market

With the data corrected, the factor test reverses too:

| Period | Mean monthly top-minus-bottom decile spread | Months positive |
|---|---|---|
| 2015–2023 | +2.06% | 61% |
| 2024–2026 | +1.91% | 58% |

**The momentum factor is healthy** — barely changed. The earlier "factor halved" reading was contaminated by phantom corporate-action crashes. So the 2024–26 losses are an implementation failure, not a market regime change.

The trade autopsy locates it precisely. Across all three years examined, **100% of exits were trailing stops and 0% were the intended rank-based exits** — the momentum thesis never got a chance to play out. Stop sensitivity over 2024–26 confirms it:

| ATR stop multiple | Return | Stops fired |
|---|---|---|
| 3× | −15.5% | 30 |
| 5× | −19.4% | 21 |
| 8× | −17.4% | 13 |
| **off** | **−0.3%** | **0** |

Removing the stop improves the result by **15–19 percentage points**. An ATR trailing stop fires on ordinary noise in volatile Indian names, converting drawdowns into realised losses and taxable events, exactly when momentum needs to let winners run. Hit rates of 8–20% with roughly symmetric win/loss sizes are the signature: the strategy was being stopped out of everything.

**Next experiment, properly specified:** momentum with rank-based exits only (no trailing stop), or a stop wide enough that the rank exit fires first. That is a hypothesis about *our construction*, which the ablation framework can test cleanly.

### Permutation tests (steps 2 and 4) — the decisive negative result

Previously only steps 1 and 3 of the four-step framework were being run, with the Deflated Sharpe substituting for the permutation tests. DSR is parametric; permutation assumes nothing and re-runs the entire procedure including the optimiser. Both now implemented.

**The null:** returns permuted *across symbols within each day*. This preserves every daily market return (benchmark untouched) and each day's cross-sectional spread, while destroying persistence in which stock earns which return. Shuffling *dates* instead would be wrong and dangerous — it preserves each stock's total return, so a momentum ranker still finds the eventual winners and the null quietly contains look-ahead.

| Step | Result | |
|---|---|---|
| 1. In-sample excellence | best Sharpe **+0.998** on the grid | — |
| 2. In-sample permutation | **p = 0.935** | **FAIL** |
| 3. Walk-forward | OOS Sharpe **+0.790** | — |
| 4. Walk-forward permutation | **p = 0.375** | **FAIL** |

Step 2 is not a marginal miss. The real data's optimised Sharpe (+0.998) was **beaten by 93% of permutations** (permuted mean +1.221, best +1.517). The optimiser does *better* on structureless data than on real data.

**The obvious objection — that the permutation makes the null easier — was tested and refuted.** Permuted portfolios show 17.4% annualised volatility against 16.9% real (1.03×) and −34.3% max drawdown against −35.0%. Risk is essentially identical; the entire Sharpe gap is return. Momentum selection in this universe is genuinely *worse than random selection*.

**A coherent explanation, consistent with everything else measured:** the earlier factor test found a healthy positive decile spread (+1.9%/month) across the *broad* liquid universe, but the strategy trades only the **top 60 by liquidity**. Published NSE work finds the Indian momentum premium is substantially a liquidity premium — low-turnover momentum returned ~19.4% CAGR against ~8.5% for high-turnover. If so, the signal is real but absent precisely where we are looking, and the tradability gate that makes the strategy executable is also what removes its edge. That remains a hypothesis; testing it costs more trials.

**Implication for the paper trade:** the account was started on the drawdown-reduction thesis, not a return edge. Drawdown reduction comes from the regime filter and inverse-vol weighting, which these tests do not evaluate. But the *stock-selection* component now has no supporting evidence, and should not be described as one.

### The liquidity experiment — where the momentum premium actually lives

The permutation result above left a contradiction on the table: the factor test found a healthy +1.9%/month decile spread across the broad liquid universe, while the permutation test found our selection *inside* that universe was worse than random. Both cannot be describing the same thing.

The reconciling hypothesis was already written into Phase 1 as a tension to design around: published NSE work places the Indian momentum premium substantially in low-turnover names. We trade the top 60 by turnover. If the premium lives further down the liquidity ladder, then the tradability gate that makes the strategy executable is also what removes its edge.

`scripts/experiment_liquidity_buckets.py` measures the spread separately within each liquidity bucket. It is deliberately a **measurement, not a search** — nothing is optimised, so it barely moves the trial count the Deflated Sharpe charges against us. 124 monthly observations per bucket, 2015–2026.

| Bucket by turnover rank | Avg names | Spread/month | t-stat | Months positive | Median ADV |
|---|---|---|---|---|---|
| **1-60 (what we trade)** | 56 | **+1.26%** | **+1.83** | 56% | ₹296.7 cr |
| 61-150 | 84 | +1.14% | +2.04 | 60% | ₹98.3 cr |
| 151-300 | 133 | +1.57% | +2.82 | 65% | ₹27.7 cr |
| 301-600 | 258 | +1.61% | +3.66 | 65% | ₹6.9 cr |
| 601-1000 | 333 | +2.18% | +5.11 | 73% | ₹1.6 cr |

**The bucket we trade is the only one whose spread is not statistically reliable.** Every other bucket clears t > 2, and both the spread and its consistency strengthen monotonically as liquidity falls. That is a structural answer to the contradiction rather than a failure of search: the signal is real, and it is absent precisely where we were looking.

**A limit of this measurement, which matters for how far down to go.** The script charges a flat 0.27% round trip — broker costs and taxes from `IndiaEquityDeliveryCosts` — and **no bid-ask spread term**. Bid-ask spread is the cost component that grows as liquidity falls, so the deeper buckets are flattered: net-of-cost return rises monotonically here (+1.23% to +2.16%/month) only because the one cost that would bend it down is missing. Participation is negligible everywhere at ₹33,333 per position (0.001% to 0.203% of median ADV), so size is not the binding constraint — spread is, and it is unmeasured. Ranks 61-300 were therefore chosen as the **nearest** reliable bucket, not the best-scoring one. Ranks 301-1000 remain untested, and testing them honestly requires a spread model this project does not yet have.

### Liquidity band test (turnover ranks 61-300)

**A note on the name.** This band is defined by median daily **turnover**, not by market capitalisation. NSE bhavcopy carries no shares-outstanding column, so market cap cannot be computed from our data at all, and every bucket in this project is therefore a turnover bucket. SEBI's large/mid/small-cap categories are market-cap ranks (1-100, 101-250, 251+); the two measures correlate but are not the same thing. An earlier draft of this section called this the "mid-cap universe", which overstated the connection. The results are unchanged — the experiment measured what it measured — but the label was wrong, and the scripts are now named `validate_liquidity_band.py` and `validate_liquidity_band_parallel.py` to match.

`scripts/validate_liquidity_band.py` re-runs the entire four-step framework with the universe moved to ranks 61-300 and **nothing else changed** — same strategy (`ResidualMomentum`), same 8-configuration grid, same windows, same costs and tax. Holding everything else fixed makes this a controlled comparison rather than another search. The benchmark is buy-and-hold on the *same* band, so the comparison is like for like: beating a basket drawn from the same liquidity band is the relevant bar, not beating the most-traded names with a thinner-traded portfolio.

3,036,443 bars, 1,589 equities, 9 walk-forward windows, ₹5,00,000, deflation charged for N=36:

| | Final | CAGR | Sharpe | Max DD |
|---|---|---|---|---|
| **strategy** | **₹3,066,921** | **+22.3%** | **+1.29** | **−29.2%** |
| buy_and_hold (same universe) | ₹1,039,779 | — | +0.51 | −57.5% |

Windows beaten: **8 of 9**. DSR **0.953**, clearing the 0.95 bar — the first configuration in this project to do so.

| Step | Result | |
|---|---|---|
| 1. In-sample excellence | best Sharpe **+1.399** | — |
| 2. In-sample permutation | **p = 0.190** | **FAIL** |
| 3. Walk-forward | OOS Sharpe **+1.286** vs benchmark +0.511 | — |
| 4. Walk-forward permutation (n=20) | **p = 0.048** | **PASS** |

### Step 4 at high resolution — 200 permutations

The n=20 result was uninformative on its own. p = 0.048 is the *floor* for 20 permutations: under the (beat+1)/(n+1) convention the smallest achievable value is 1/21 = 0.048. All it established was "0 of 20", sitting on the significance threshold by construction rather than by measurement.

`backend/engine/parallel.py` and `scripts/validate_liquidity_band_parallel.py` re-run the same test across CPU cores. Workers load their own data from parquet rather than having a three-million-row frame pickled to them, and every callable crossing the process boundary is a module-level function, because Windows spawns rather than forks.

```
real result     +1.286
permuted mean   +0.670
permuted best   +1.176
real beats      200 of 200 permutations
p-value         0.005   (again the floor: 1/201)
wall clock      51.8 min   (serial estimate 366 min, 7.1x speedup)
```

**The real result beat every permutation, and beat even the best of them — +1.286 against +1.176.** p = 0.005 is once more the resolution floor, so the true p-value is lower still. This is the strongest evidence the project has produced.

**Reproduced.** The run was repeated end to end on 8 workers instead of 14 (70.2 min, 5.3x). Different worker count means a different completion order, and the intermediate running means differ early because of it — but the final distribution is identical to three decimals: mean +0.670, best +1.176, 0 of 200 beating real, p = 0.005. That rules out order-dependence or shared state in the parallel runner, and confirms the committed post-fix code still produces +1.286.

A first attempt at this reproduction died on `BrokenProcessPool` after a worker hit `_ArrayMemoryError`: 14 workers each hold their own copy of the 3M-row frame, and a second experiment was loading a 4.5M-row frame concurrently on a 23 GB machine. Note the defect it exposed — `run_permutations` calls `fut.result()` with no exception handling, so one dead worker discards every permutation already completed.

**On the GPU question, answered by profiling rather than theory:** 84% of runtime sits in `_score`, of which `pd.to_datetime` plus `reindex` are 38% of the total. That is pandas object overhead inside a sequential day loop, not arithmetic, so a GPU would idle. The real wins were a date-aligned market model (1.7x) and CPU parallelism (7.1x).

### What still stands against the liquidity band result

Recorded now, before the headline crowds them out:

1. **Step 2 still fails, at p = 0.190.** The optimiser reaches Sharpe +1.596 on structureless data against +1.399 on real data. Step 2 has limited power over an 8-configuration grid — in-sample optimisation can reach a high Sharpe on almost anything — and step 4 measures what actually generalises. But it is a failed gate, and it is not a marginal miss.

2. **Three windows carry the entire result.** 2020-03..2021-03 (+56.4%), 2021-03..2022-03 (+57.7%) and 2023-03..2024-04 (+56.9%) produce nearly all of the 5.86x. Drop those three and the remaining six compound to **1.51x over six years, about +7.2% a year** — roughly the benchmark. The good years sit exactly where Indian mid- and small-caps had an extraordinary run — genuine market-cap categories this time, describing the market rather than our universe. That is a statement about the sample, and this is the only sample there is.

3. **This is a backtest.** The clean out-of-sample data has now been used twice. The only genuinely untouched evidence remaining is forward time.

4. **Two of the last three "improvements" were bugs that flattered the results** — the silent-leverage ablation and the market-model alignment error, each of which reversed a conclusion once found. The 200-permutation run is the check that does not care what the author wants to be true, which is why it was worth 52 minutes.

**Consequence for the paper trade.** The running paper account trades `PatientMomentum` on the **top-60** universe — the one bucket with no reliable spread. Its configuration is frozen deliberately: editing a running forward test destroys the only untouched evidence it exists to produce. The correct move is a **second, separate paper account** on the turnover-rank 61-300 band, running alongside the first rather than replacing it.

### Historical validation (2005-2014) — the frozen configuration does not survive 2008

The whole project had been validated on 2015-2026. That decade contains COVID — a fast crash and a fast recovery — and nothing resembling 2008, a fourteen-month grind down. Roughly 2,500 trading days of pre-2015 bhavcopy were ingested (see `india_historical.py` for the era-aware cost and tax model) and the **frozen configuration was run once, unchanged**. One trial, nothing searched.

| Arm | Final | CAGR | Sharpe | Max DD | 2008-01..2009-03 |
|---|---|---|---|---|---|
| strategy (as running) | ₹348,110 | −5.0% | −0.12 | **−68.9%** | −66.0% |
| + regime filter | ₹311,816 | −6.5% | −0.22 | −68.8% | −65.5% |
| + regime **liquidation** | ₹447,907 | −1.6% | −0.25 | **−20.1%** | +0.0% |
| **buy-and-hold (same band)** | **₹709,202** | **+5.1%** | **+0.32** | −65.8% | −58.4% |

**Nothing beat buy-and-hold.** Every arm lost money over the decade while the benchmark compounded at +5.1%.

**The −68.9% drawdown reproduces the −70.5% figure this plan has cited since it was written.** That number was the stated justification for the regime filter. It has now been measured on our own configuration, and it is real. The worst ever seen on 2015-2026 was −26.1%.

**The regime filter does not do the job this plan assigned it.** It fired correctly — risk-on only 5% of days through the crash — but `target = [...] if regime_ok else []` only empties the BUY list. A book already holding 15 names rides the decline all the way down; the gate merely stops it adding. It bought 0.1pp of drawdown protection. The claim that it is "the entire justification... an unfiltered momentum book would have been unholdable through 2008" was wrong about the mechanism: as built, it is an entry filter wearing a crash defence's justification. This was invisible on 2015-2026, where the one sharp drawdown was fast enough that an entry gate looked adequate.

**Making risk-off liquidate works, and costs what such filters always cost.** Checked daily rather than on the quarterly rebalance clock — five decision points in a fourteen-month decline would arrive months late — it cuts the drawdown to −20.1% and sits out 2008 entirely. It also missed the recovery: **+59.1% for the unfiltered arm in the 2009 window against −0.7%** for this one, and it held cash through all of 2011-12. Getting out keeps you out.

**Two engine bugs this run exposed, both mine, both caught by impossible numbers.**

1. *The benchmark bought nothing for four windows*, sitting at exactly ₹500,000 through a crash that took Indian equities down about 60%. The engine rejects positions costing more than `max_cost_pct` (0.5%) — a 2026 rule aimed at the flat ₹13 DP charge. In 2008 the *proportional* cost floor alone was 1.45%, so no quantity at any price could satisfy it. The two code paths then disagreed about the unsatisfiable constraint: `min_quantity_for_cost_budget` returns 0, so the engine's `shares < min_shares` test never fires and a strategy trades freely, while `BuyAndHold.affordable_symbols` tests cost directly and buys nothing. This is the Phase 4 frozen-benchmark failure, reintroduced through era-aware costs. `cost_budget_for()` now sets the budget to the era's own achievable floor plus 0.3pp — 1.75% in 2008, 0.52% today.
2. *The regime liquidation initially ran on the rebalance clock*, which would have tested a quarterly crash defence and proved nothing.

**Caveat that cuts against the result being even this good:** 411 pre-2011 symbols were dropped for unmappable ISIN — 8.6% of pre-2011 turnover — so the historical universe tilts toward survivors. Reality was probably worse.

### CORRECTION — the 2005-2014 results above were measured with a cold lookback

The historical run and the regime-liquidation run both omitted the warmup tail that `WalkForward` applies, so the strategy began every test window with a cold 126-day lookback and could not rank anything for months. The bug was found in `validate_deep_band.py` and **not propagated** to `validate_historical.py` at the time. Corrected numbers:

| Arm | reported | corrected |
|---|---|---|
| strategy final | ₹348,110 | **₹803,205** |
| strategy CAGR | −5.0% | **+7.0%** |
| strategy Sharpe | −0.12 | **+0.42** |
| strategy max DD | −68.9% | −59.6% |
| strategy, 2008-01..2009-03 | −66.0% | −53.7% |
| + regime filter, max DD | −68.8% | **−39.2%** |
| + regime filter, 2008 | −65.5% | **−36.7%** |
| + regime liquidation, max DD | −20.1% | −14.2% |
| benchmark | ₹709,202, +5.1%, −65.8% | ₹708,099, +5.1%, −65.8% |

**Three stated conclusions were wrong and are withdrawn:**

1. *"The frozen configuration does not survive 2008"* and *"nothing beat buy-and-hold"* — it did beat its benchmark, ₹803,205 against ₹708,099, Sharpe +0.42 against +0.32.
2. *"It falls further than buy-and-hold in the crash"* — it fell −53.7% against −58.4%. It protected, mildly.
3. *"The regime filter is an entry gate wearing a crash defence's justification"* — with the lookback warm it cuts drawdown by 20 points and the crash loss by 17. The mechanism described (it empties the buy list, it does not sell) is still accurate; the conclusion that this makes it useless was not.

**What survives the correction:** the drawdown is far worse than the decade the strategy was designed on (−59.6% against −26.1%), risk-off liquidation cuts it hardest (−14.2%) at a real cost in return, and the edge over the benchmark is modest at +1.9pp CAGR.

**The lesson is about the harness, not the strategy.** This is the fourth time a simulation detail has reversed a finding here, and the third time a control arm or a failed reproduction was the only thing that caught it. A conclusion from this project should not be trusted until the same configuration has been reproduced by a second code path.

### Volatility targeting — an explicit risk target, from Carver Table 25

Portfolio volatility was previously an accident of holding 15 inverse-vol-weighted names: 14.1% annualised over 2015-2026, never chosen. Carver's Table 25 puts the right target at 10% for a realistic 0.40 Sharpe with negative skew (ours is −0.49, with excess kurtosis +4.2). Target pre-registered at 10% from that table, not tuned here. It can only de-risk: Indian cash equity delivery cannot be levered, so the multiplier is capped at 1.0.

| Period | Arm | Final | CAGR | Vol | Sharpe | Max DD |
|---|---|---|---|---|---|---|
| 2015-2026 | no target | ₹2,152,835 | +17.6% | 14.1% | +1.22 | −26.3% |
| 2015-2026 | target 10% | ₹1,405,841 | +12.2% | 11.6% | +1.05 | −26.6% |
| 2005-2014 | no target | ₹892,708 | +8.6% | 21.8% | +0.49 | −59.2% |
| 2005-2014 | target 10% | ₹713,672 | +5.2% | 13.2% | +0.45 | −34.9% |

**It does what risk targeting is supposed to do, and nothing more.** In the calm decade it costs 5.4pp of return, cuts volatility by 2.5 points, and does not improve the drawdown at all. In the turbulent decade it cuts volatility from 21.8% to 13.2% and the drawdown from −59.2% to **−34.9%** — 24 points — for 3.4pp of return, with Sharpe essentially unchanged (+0.49 to +0.45).

That asymmetry is the point: it is insurance, priced in foregone return, that pays out only in the regime the strategy has never traded live. Whether the premium is worth paying is a risk-appetite decision rather than a return-optimisation one, which is exactly how Carver frames it. **It is not enabled on any running account.**

### A learned ranker (gradient-boosted trees) — loses to the one hand-chosen number

Built because a model can weigh many signals at once and, retrained each window, can change HOW it decides rather than only what it picks. `MLMomentum` subclasses `ResidualMomentum` and overrides **only** `_rank`, so sizing, exits, the rank buffer, weighting, the rebalance clock, costs and taxes are the same code. The universe, windows and capital are identical. Hyperparameters fixed a priori, not tuned: one configuration, one trial.

| | Final | CAGR | Sharpe | Max DD |
|---|---|---|---|---|
| ResidualMomentum | ₹2,075,856 | +17.1% | **+1.21** | −28.8% |
| ML ranker | ₹1,276,336 | +11.0% | +0.80 | −56.1% |
| buy-and-hold (control) | ₹1,191,017 | +10.1% | +0.60 | −57.5% |

It beats its control by +0.194 Sharpe and loses to the hand-chosen rule by **−0.414**. DSR 0.542 at N=50, failing.

**The diagnostics say why, and they are the useful part.** Out-of-sample information coefficient — rank correlation between prediction and realised forward rank — averages **+0.0184**, positive in 8 of 9 windows, t = +1.21. So the model has weak, genuinely non-zero predictive power. It is simply far too noisy to trade: an IC that small, at quarterly turnover, is buried by 0.42% round-trip costs and 20% STCG.

**What it learned is the second half of the answer.** By split count the model leans hardest on `turnover_rank` (693), `vol_63` (538), `mom_252` (510) and `skew_63` (410). Momentum features are present but not dominant — the model mostly rediscovered **liquidity and volatility**, which are characteristics the universe construction already fixes by trading a defined turnover band.

The drawdown confirms it: −56.1% against the control's −57.5% and ResidualMomentum's −28.8%. The learned book carries market-like risk without market-beating return, because ranking on generic characteristics produces a portfolio that looks like the market. `ResidualMomentum`'s single number wins precisely because it is *not* generic — stripping market beta isolates idiosyncratic trend, which is the thing worth concentrating in.

**Seventh refinement, seventh null.** Conviction weighting, wider exits, absolute-only exits, tax-aware exits, the regime filter as an entry gate, the deep band, and now a learned ranker. The pattern is consistent enough to be a finding in itself: on twelve price-derived features over eleven years, added model capacity does not buy anything.

Published context: best-in-class monthly out-of-sample R² for cross-sectional equity ML is 0.3–0.4%, from far richer data than price-derived features. This result is not anomalous.

### The residual label — predictive power triples, and cannot be harvested

The first model was diagnosed as learning generic characteristics because its target rewarded them: raw forward return is substantially explained by beta, volatility and liquidity. So the label was changed to **residual forward return** — `fwd_stock − beta × fwd_market`, with beta estimated on a trailing window. One change, one trial.

The diagnosis was correct and the fix worked, in the direction predicted:

| Label | Sharpe | Max DD | Out-of-sample IC | t-stat |
|---|---|---|---|---|
| raw forward return | +0.80 | −56.1% | +0.0184 | +1.21 |
| **residual forward return** | +0.84 | −49.9% | **+0.0590** | **+3.94** |
| *ResidualMomentum (the rule it must beat)* | *+1.21* | *−28.8%* | — | — |

**Predictive power tripled and became statistically significant.** An IC of +0.059 at t = +3.94 is a respectable factor IC — this model genuinely forecasts cross-sectional residual returns.

**And the portfolio barely moved: +0.80 to +0.84 Sharpe.** That gap is the result worth keeping.

**Why: a long-only cash-equity book earns RAW returns, not residual ones.** Harvesting residual return requires shorting the market against each position, and Indian cash equity cannot hold overnight shorts (noted in this plan's opening constraints). The model is predicting a quantity the portfolio structurally cannot collect. A stock can have excellent residual performance and still lose money, and the account holds the stock.

Feature importances are identical across both runs — 31% momentum, 60% generic — confirming this is not a modelling artifact. The model used the same inputs; only the target changed.

**What this settles.** The binding constraint is not the model, the hyperparameters, or the label. It is that the *harvestable* signal in price-derived features is small, while the part that is strongly predictable is unavailable to a long-only account. `ResidualMomentum` beats the model not by predicting better but by using residual momentum as a SELECTION signal while still earning raw returns — the correlation between the two is what it monetises.

Eighth refinement, eighth null. Running the model on more data or with more capacity would raise IC further and change nothing about harvestability.

### Phase 5 — Dashboard *(your requirement #4a)*
React UI: tradability explorer, run browser, equity-curve overlay for N runs, side-by-side metrics table, trade-level drill-down, parameter-sensitivity heatmaps.

### Phase 6 — Explain chatbot *(your requirement #4b)* — **zero cost, no API calls**

**Hard constraint: nothing in this phase makes a paid API call.** No Anthropic calls, no metered service of any kind. Cost is ₹0 by construction, not by budget discipline.

The design that makes this work is putting the **query layer, not the model, at the centre**. `chat/tools.py` is a set of plain Python functions returning structured data:

`list_runs` · `get_run_metrics` · `get_trades` · `compare_runs` · `explain_signal(date, symbol)` (returns the stored decision record) · `get_tradability(symbol, date)` · `get_regime(date)`

These are the whole substance of the chatbot. Every answer is a rendering of what they return. Two free front-ends sit on top:

**Tier 0 — deterministic intent router (default, always available).** A pattern matcher over a fixed grammar of question types maps a question to one tool call plus a template. This is not a degraded fallback — for the most important question it is *strictly better* than any LLM:

> *"Why did you buy TITAN on 14 March?"* → renders the stored decision record as a table: every rule, its inputs, its threshold, its boolean outcome, the regime state, the ATR stop distance, the computed share count, and the constraint that bound the size.

Because Phase 2 already logs all of that, the answer is exact, complete, instant, and **cannot hallucinate**. A language model asked the same question would paraphrase this table less reliably. Covering "why this trade", "what were the metrics", "how do these two runs differ", and "why was this stock rejected" gets you most of requirement #4 with no model at all.

**Tier 1 — optional local LLM (Ollama), for open-ended questions.** Your machine handles this comfortably: **RTX 5070 Laptop, 8GB VRAM, 23GB RAM, Ryzen 9 8940HX** — a quantised 7–8B model (Qwen2.5-7B-Instruct or Llama 3.1 8B) runs fully on GPU. It handles "why did this underperform in 2022?" by calling the same tools and summarising. Runs locally, costs nothing, works offline. Ollama is not currently installed; it's a Phase 6 setup step, and it stays **optional** — Tier 0 must be fully functional without it.

**A paid-provider adapter is left as a stub behind a disabled-by-default config flag.** Because the tools layer is provider-agnostic, switching one on later is a config change rather than a rewrite. It ships **off**, with no key wired and no code path that can reach it.

- **Exit criteria:** every question type in the Tier 0 grammar answers correctly from stored data with Ollama uninstalled, and a network-disabled run of the test suite passes — proving no outbound paid call exists

### Phase 7 — Paper trading
Kite Connect subscription starts here. `PaperBroker` consumes live Kite quotes and simulates fills with the same cost model. Runs the identical strategy code. Forward-test for a meaningful period — this is the only genuinely out-of-sample evidence you will ever get.

### Phase 8 — Live, human-approved
`KiteBroker`. Proposed orders surface in the UI with size and rationale; nothing transmits without explicit confirmation. Kill switch, daily loss limit, position reconciliation against broker state on every cycle. Start with a small fraction of the capital you eventually intend to deploy.

---

## Verification

- **Unit:** cost model against Zerodha's published brokerage calculator for several trade sizes; indicator outputs against TA-Lib; sizer respects every constraint including integer rounding
- **Engine integrity:** buy-and-hold replication (Phase 2 exit criteria); a deliberately look-ahead-biased strategy must be *impossible* to express through `BarView` — write a test asserting that accessing future bars raises
- **Data integrity:** known corporate action reconciles; universe on a past date contains names since delisted (proves point-in-time works and survivorship bias is absent)
- **End-to-end:** `scripts/run_backtest.py --strategy ma_crossover --from 2018-01-01 --to 2024-12-31` produces a persisted run; it appears in the dashboard; the chatbot explains any trade in it from the stored decision record
- **Zero-cost guarantee:** the full suite passes with outbound network disabled, and a grep for paid-provider SDK imports outside the disabled stub returns nothing
- **Pre-live:** paper-trading fills reconcile against what the backtest engine would have produced on the same bars

---

## Honest expectations

The research you commissioned already says this, and it is worth restating as an engineering constraint rather than a caveat: best-in-class academic ML models achieve ~0.33–0.40% monthly out-of-sample R². Most retail strategies do not beat buy-and-hold after costs.

For your specific situation the bar is higher than that research implies, because it was written from US assumptions. Stacked up, a ₹50,000 Indian swing account faces:

1. **~0.5% per round trip** in costs, of which a fixed ₹13 DP charge is the worst part at small size
2. **20% STCG vs the benchmark's 12.5% LTCG**, plus the loss of tax deferral
3. **4–5 concurrent positions maximum**, so very little diversification
4. A momentum edge that the Indian evidence suggests **lives in the less-liquid names you are filtering out**

None of these are reasons not to build it. They are reasons to build the measurement apparatus (Phases 0–4) properly before believing any result, which is exactly how this plan is sequenced.

The most likely outcome of Phase 3 is that the first few strategies fail to beat buy-and-hold after costs. **That is the system working correctly** — it is telling you something true before you risk money on it. The value of Phases 0–4 is that they make that verdict trustworthy. Build them properly and the answer, whichever way it goes, is worth having.
