# Decision rule for the paper-trading accounts

Written 2026-09-09, **before** the forward test has produced anything readable.
That timing is the whole point. Deciding what counts as success after seeing
the results is how you end up believing whichever story the numbers happen to
tell, and this project has spent 49 trials building an apparatus specifically
to avoid that.

---

## 1. The question I originally wanted to ask, and why it cannot be answered

The obvious rule is: *does `band_residual` beat `band_hold` by enough to
justify real money?*

That question is unanswerable on any horizon you would accept. Measured on
this project's own walk-forward equity curves, 2015-2026:

```
daily-return correlation, strategy vs its own control : 0.695
Sharpe advantage                                      : +0.609
```

Carver's Table 6 (*Systematic Trading*, ch. 3) gives the years of data needed
to be reasonably certain one rule beats another, by Sharpe advantage and
correlation:

| SR advantage | corr −1.0 | 0.0 | 0.5 | 0.8 | 0.95 |
|---|---|---|---|---|---|
| 0.25 | 46 | 45 | 40 | 32 | 10 |
| 0.5 | 41 | 37 | 25 | **10** | 3 |
| 1.0 | 23 | 11 | 8 | 2.5 | 0.5 |

At an advantage of ~0.5-0.6 and correlation 0.695, we sit between the 25-year
and 10-year cells: **roughly 12-15 years of forward data** before the
comparison becomes statistically meaningful.

His Table 5 says the same thing from the other direction — confirming that a
strategy with a true Sharpe of 0.5 is even *profitable* takes about 20 years.

**So: this strategy cannot be proven better than holding, in practice, ever.**
That is a statistical limit, not a flaw in the build. A 12-month test with a
5-percentage-point threshold — which is what I first proposed — would have
been answered almost entirely by noise, and whichever way it came out I would
have believed it.

---

## 2. What replaces it: a stopping rule

Proof and disproof are not symmetric here. Confirming a modest edge takes
decades; **detecting that something is broken or clearly worse takes months.**
So the rule below is a stopping rule and a behaviour check, not a go-live
test.

### 2a. Stop conditions — any one triggers a full review

Reviewing means investigating and deciding, not automatic abandonment. But
none of these may be ignored or explained away without writing down why.

1. **Excess return below −10pp annualised** against its own control, over any
   rolling 12-month window. A modest edge is undetectable; a strategy losing
   badly to holding the same names is not.
2. **Drawdown beyond −40%.** The 2015-2026 backtest never exceeded −26.1%. A
   live drawdown past 40% means live risk exceeds modelled risk, which is a
   statement about the model rather than about the market.
3. **Realised round-trip cost more than 50% above the model.** The cost model
   is the single most load-bearing assumption in this project. If it is wrong,
   every result is wrong.
4. **Turnover more than 50% off the backtest.** Quarterly rebalancing with a
   rank buffer implies a predictable trade count. A large deviation means the
   live strategy is not the tested strategy.
5. **Any account diverging from its own logged intent** — holding a different
   number of positions than signalled, orders not filling as modelled, state
   drifting between runs.

### 2b. Behaviour checks at 6 months — does live match the backtest?

These are answerable on short data because they are about *mechanics*, not
about edge. Record and compare against backtest expectations:

- trades per year per account
- realised cost per round trip against `IndiaEquityDeliveryCosts`
- realised slippage against the modelled 0.1% (and the measured band spreads:
  0.16% for 61-300, 0.34% for 601-1000)
- average and maximum position count
- exposure, cash drag
- realised STCG drag against the 20% assumption

A mismatch here is worth more than any return number, because it is
diagnosable and fixable.

---

## 3. What would actually justify real capital

Not a statistical result — that is unavailable. A judgement, made under
irreducible uncertainty, with these preconditions:

1. **At least 12 months elapsed** with no stop condition triggered.
2. **Behaviour matches the backtest** on the section 2b checks.
3. **Excess over control is not negative.** Weak evidence, deliberately — it
   is a floor, not a proof.
4. **You accept explicitly that the edge is unproven and may be zero.**
5. **Capital sized so the demonstrated downside is survivable.** Not the
   −26.1% from 2015-2026, but the **−68.9%** this exact configuration produced
   over 2005-2014. Size so that losing 69% of the committed amount is an
   acceptable outcome, because it has already happened once in the sample.

At ₹50,000 that is a ₹34,500 loss. That is the number to be comfortable with
before starting, not the expected return.

---

## 4. External calibration on what is achievable

Carver's realistic Sharpe ratios (ch. 2), worth holding against our own
numbers:

| What | Realistic Sharpe |
|---|---|
| Single equity | 0.15 |
| 20+ diversified equities, one country | 0.20 |
| Global equities | 0.25 |
| Multi-asset static portfolio | 0.40 |
| Dynamic rules, single asset class | 0.40 |

And: *"SR consistently greater than 1.0 are rarely achieved, even by
sophisticated institutional investors. Analysis of returns from a group of
systematic hedge funds shows that none could sustain a Sharpe ratio above 1.0
for more than a few years."*

Our headline was **+1.286**, on a single asset class, in one country, long
only. That is roughly three times what an experienced practitioner considers
achievable for this shape of strategy, and above what he says institutions
sustain.

That is not proof it is wrong. It is a strong prior that it is, and it is
consistent with everything else measured here: the same configuration returned
**−0.12** across 2005-2014.

---

## 5. What this project should stop doing

Carver's Table 12 gives weight adjustments for assets with different Sharpe
ratios. Column C covers the case of **less than ten years of data**, and every
entry in it is **1.0** — no adjustment, for any Sharpe difference at all. His
position is that with short data you should not act on measured performance
differences between rules, because they carry no information.

This project has run 49 trials over roughly eleven years of data and
repeatedly chosen between variants on Sharpe differences of 0.03 to 0.28.
Table 6 says distinguishing a 0.25 advantage at zero correlation takes 45
years. Those choices were not measurements; they were noise, and the ones
reported as null results were the honest outcome.

Concretely, from here:

- **No more strategy variants on this data.** At N=49 the deflation bar is
  above anything a marginal winner could clear.
- **No re-tuning of a running account.** Ever. It converts the only untouched
  evidence into an in-sample fit.
- **Infrastructure and diagnostics are still fair game** — they cost no
  trials. Fundamentals ingest, the explain layer, the weekly report.

---

## 6. Review schedule

- **Weekly** — automated report, no decisions.
- **6 months (2027-03)** — behaviour checks only. No return judgement.
- **12 months (2027-09)** — first point at which section 3 may be applied.
- **Any time** — a section 2a stop condition triggers an immediate review.

The accounts run on a Windows scheduled task at 19:30 each weekday. Missed
days are caught up automatically; gaps do not need manual repair.
