# What Carver's *Systematic Trading* says about this project

Read 2026-09-09 against the state of the system at commit `fc0ff6c`. Robert
Carver ran AHL's multi-billion fixed-income portfolio and now trades his own
book; the book is about framework design, risk and fitting rather than about
finding signals, which makes it a better fit for where this project actually
is than a book of strategies would be.

Below: what it confirms, what it contradicts, and the one structural gap it
exposes. Every number attributed to this project was re-measured for this
document rather than quoted from memory.

---

## 1. Where our headline number sits against reality — it doesn't

Carver's realistic Sharpe ratios (ch. 2):

| Portfolio | Realistic SR |
|---|---|
| Single equity | 0.15 |
| 20+ diversified equities, one country | 0.20 |
| Global equities | 0.25 |
| Multi-asset static | 0.40 |
| Dynamic rules, single asset class | 0.40 |

And directly: *SR consistently greater than 1.0 are rarely achieved even by
sophisticated institutional investors,* with no systematic hedge fund in his
sample sustaining above 1.0 for more than a few years.

Our headline was **+1.286**, from a long-only, single-country, single-asset-class
strategy — roughly three times what he considers achievable for this shape.
He attributes numbers like that to over-fitting.

This is not proof our number is wrong, but it is an independent prior that it
is, arrived at without seeing our data. It agrees with what our own data said
later: the same configuration returned **−0.12** across 2005-2014.

## 2. Why the forward test cannot settle the question — measured, not assumed

Carver's Table 6 gives years of data needed to be confident one rule beats
another, by Sharpe advantage and correlation. Measured on our own
walk-forward curves: correlation **0.695**, Sharpe advantage **+0.609**. That
lands between his 25-year and 10-year cells — **12-15 years**.

His Table 5 says confirming a true Sharpe of 0.5 is even *profitable* takes
about 20 years.

This is what turned the decision rule from a go-live test into a stopping
rule. See [DECISION_RULE.md](DECISION_RULE.md).

## 3. The 49 trials were mostly noise, and he quantifies why

Table 12 gives weight adjustments for assets with different Sharpe ratios.
**Column C — less than ten years of data — is 1.0 for every single row.** No
adjustment, for any Sharpe difference, because at that sample size measured
differences carry no information.

Table 6 puts numbers on it: distinguishing a 0.25 Sharpe advantage between
uncorrelated rules takes 45 years.

This project chose between variants on differences of 0.03 to 0.28 across
eleven years, 49 times. The ablations reported as null results were the
honest outcome; the ones that looked like wins were the ones to distrust.

Table 4 is the one place we scrape through. Testing a pool of 50 rules on 10
years of data, he puts the Sharpe cutoff at **1.0** for a 5% chance of picking
a bad rule. Our +1.21 clears that — narrowly, on the decade it was fitted to,
and not on the other decade.

## 4. What the book confirms we got right

**Rank exits over stops.** He is scathing about stop-losses inside systematic
rules for the same reason our post-mortem found — they fire on noise. Our
measurement (100% of exits were stops, 0% were rank exits; removing them was
worth 15-19 points) is an independent rediscovery of his position.

**Benchmark-relative comparison.** *"Be careful of focusing on outright
performance, rather than returns relative to benchmarks... many asset classes
have done extraordinarily well over the last 40 years."* Every verdict here
is against a same-universe buy-and-hold control, which is exactly this.

**Equal weight beating optimised weights.** Our weighting ablation found plain
equal weight beat both inverse-vol and conviction weighting. His entire
Chapter 4 is about how optimisers produce garbage and why he replaced them
with hand-built weights.

**Turnover is not our problem.** His speed limit: never pay more than a third
of expected profits in costs, which at a realistic 0.40 SR means never more
than **0.13 SR annually**. Measured over our 9 out-of-sample years:

```
round trips per position slot per year : 1.49
cost per leg                           : 0.123% of notional
standardised cost                      : 0.0174 SR per round trip
annual cost in Sharpe terms            : 0.026 SR
Carver's limit                         : 0.13 SR
headroom                               : 5.0x
```

The quarterly rebalance, the rank buffer and the removal of stops — all
decisions this project reached from its own post-mortems — put it comfortably
inside his limit. This is the first external check the system has passed
cleanly.

## 5. The one structural gap: there is no risk target

This is the substantive thing the book has that we do not.

Carver treats the overall risk level as *the* most important design decision,
set as an explicit **percentage volatility target** — a desired annualised
standard deviation of portfolio returns — with positions sized to hit it.

We have no such mechanism. Our portfolio volatility is whatever falls out of
holding 15 inverse-vol-weighted names. Measured over 2015-2026:

```
annualised volatility : 14.1%
skew                  : -0.49
excess kurtosis       : +4.2
worst day             : -5.70%
max drawdown          : -26.3%   (and -68.9% over 2005-2014)
```

His Table 25 recommends a target from realistic Sharpe and skew, using
half-Kelly and halving again for negative skew:

| Realistic SR | skew > 0 | negative skew |
|---|---|---|
| 0.25 | 12% | 6% |
| 0.40 | 20% | **10%** |
| 0.50 | 25% | 12% |
| 1.0+ | 50% | 25% |

Our skew is negative (−0.49) with fat tails (+4.2). Taking his realistic
Sharpe of 0.40 rather than our backtested 1.21, the recommended target is
**10%**. We run **14.1%** — about 1.4x more risk than the framework says this
strategy's realistic edge justifies.

That is not catastrophic. He devotes pages to systems running implied targets
of 160%, and ours is nothing like that. But two things matter:

1. **It is untargeted.** 14.1% is an accident of the position count and the
   weighting scheme, not a decision. In a higher-volatility regime it will
   drift upward exactly when it should not, which is a plausible contributor
   to the −68.9% in 2005-2014.
2. **It assumes an edge we do not have.** Running 14.1% is the correct target
   for a strategy with a realistic Sharpe near 0.6. Ours is 1.21 in-sample
   and −0.12 out of it.

**Adding a volatility target is a risk decision, not a signal search — it
costs no trials.** It is the clearest actionable item the book produces.

## 6. His philosophy, which is the opposite of what we did

Carver does not select rules on performance at all. He picks a small number of
rules from ideas, chooses variations by *behaviour* (trading speed,
correlation) rather than returns, then allocates weights accounting for Sharpe
uncertainty — so a weak rule receives a small weight rather than being
discarded. Returns data is reserved for weighting, never for selection.

This project did the opposite: a grid search per walk-forward window, keeping
the best. That is the approach his "50 model kid" anecdote exists to mock, and
the Deflated Sharpe is the machinery we built to punish ourselves for it.

Worth noting, in fairness: our procedure was honest about the cost — every
trial was counted and charged. His procedure avoids incurring the cost at all.

---

## Actionable, in order

1. **Add an explicit volatility target** (section 5). No trials. The one real
   structural gap.
2. **Stop selecting on performance differences** below what the data can
   resolve (section 3). Already written into the decision rule.
3. **Hold the achievable-Sharpe numbers** (section 1) as the prior when
   reading any future backtest from this project.

Nothing here suggests a new strategy to test, which is the correct outcome
given where the evidence sits.
