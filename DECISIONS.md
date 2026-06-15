# DECISIONS.md — Design Choices & Assumptions

All non-obvious decisions made during implementation are recorded here so they can be revisited.

---

## D-01: Bayesian Model — Full Refit vs. Warm Start

**Decision:** Full refit on every `make update` call in MVP.

**Reasoning:** The WC2026 group stage has at most ~48 group matches before knockout begins. Each full refit on a modern laptop takes 2–5 minutes with `draws=1000, chains=2`. This is acceptable for a manually-triggered update. Warm start (using previous posterior as prior) is architecturally appealing but requires serializing PyMC trace objects across runs, which complicates the implementation significantly.

**Future revisit:** If refit time exceeds 10 minutes (unlikely with PyMC's NUTS sampler on this dataset size), implement warm start by saving the trace to `netCDF4` via ArviZ and loading it as the starting point.

---

## D-02: Historical Training Cutoff

**Decision:** Use all matches from 1872–present for fitting, with exponential time-decay (half-life = 2 years) rather than a hard cutoff.

**Reasoning:** A hard cutoff (e.g., "only last 5 years") discards useful signal about team relative strengths. Time-decay allows the model to use all history while de-emphasizing old matches. For the holdout/evaluation split, we use the last 2 years as holdout — not excluded from training in the primary model, but used only for metric reporting.

---

## D-03: match_id Format

**Decision:** `f"{date}_{normalize(home)}_{normalize(away)}"` where normalize strips accents, lowercases, and replaces non-alphanumeric with `_`.

**Reasoning:** This is source-agnostic and deterministic. Football-data.org and Wikipedia use different team names ("Ivory Coast" vs "Côte d'Ivoire") — normalization handles this. The format is human-readable.

**Edge case:** If the same two teams play twice on the same date (theoretically possible in some tournaments), match_ids would collide. This has never occurred in the historical dataset (49k+ matches) and is a known limitation.

---

## D-04: WC2026 Group D Composition

**Decision:** Turkey, Australia, Paraguay, United States (as stated in build spec).

**Note:** The spec explicitly warns: "Avusturya DEĞİL" (not Austria). Confirmed: Turkey is in Group D with Australia, Paraguay, and USA. This is hardcoded in `sim/group_sim.py` and `sim/turkey.py`.

---

## D-05: Score Matrix Max Goals

**Decision:** `MAX_GOALS = 8` (i.e., 9×9 matrix covering 0–8 goals per team).

**Reasoning:** The probability of a team scoring 9+ goals in a single match is vanishingly small (<0.01% for average teams). Truncating at 8 captures >99.9% of the probability mass in all realistic scenarios.

---

## D-06: Probability Time Series Append Strategy

**Decision:** Each `make update` appends one row to `probability_timeseries.parquet` using the snapshot timestamp as key. Duplicate timestamps are deduplicated on append (same run twice = idempotent).

**Reasoning:** Parquet supports efficient append-and-dedup patterns. This builds the time series automatically as the tournament progresses.

---

## D-07: Knockout Probabilities in MVP

**Decision:** MVP uses a naive power-law decay to estimate Turkey's knockout-stage probabilities. Full bracket simulation is Phase 2.

**Reasoning:** WC2026 uses a 32-team bracket from 12 groups of 4 (top 2 + 8 best third-place). Correctly modeling the bracket requires knowing all 48 group compositions and the seeding rules — data not yet available before the tournament starts. The placeholder gives a rough estimate; replace with `bracket_sim.py` in Phase 2.

---

## D-08: xG Feature Disabled by Default

**Decision:** `ENABLE_XG=false` in MVP.

**Reasoning:** soccerdata's FotMob/Sofascore scrapers are fragile and dependent on undocumented APIs. Failing to fetch xG should never block the core prediction pipeline. The Bayesian Poisson model performs well without xG (Brier on historical holdout is the ground truth).

---

## D-09: Streamlit App Reads Only from `data/outputs/`

**Decision:** The web app never runs models or fetchers — it reads pre-computed parquet/JSON files from `data/outputs/`.

**Reasoning:** Streamlit Cloud has resource limits and caching inconsistencies. Pre-computing outputs makes the app fast, stateless, and deployable anywhere. The separation also ensures model re-runs don't block the UI.

---

## D-10: Team Name Normalization Across Sources

**Decision:** Match IDs use a normalized form, but the display names in predictions use the source's original string (football-data.org names). Turkey is stored as "Turkey" (not "Türkiye") to match historical dataset and football-data.org.

**Reasoning:** Keeping display names source-consistent avoids confusion. The normalized match_id handles cross-source dedup without changing display names.

---

## D-11: LightGBM Training Window

**Decision:** LightGBM is trained on the same 8-year window as the Bayesian model (last 8 years of historical data). The last 2 years of that window are held out for Brier-based ensemble weight optimisation and isotonic calibration.

**Reasoning:** Using the same window ensures both models see identical data, making their holdout metrics directly comparable. An 8-year window gives ~4,600 training samples and ~1,150 holdout samples — sufficient for LightGBM's 17 features without overfitting. The early-stopping validation set (last 20% of training = last ~1.6 years before holdout) provides a leakage-free stopping criterion.

**Observed result:** LightGBM alone (54 trees, early-stopped) underperforms Bayesian on this dataset (Brier 0.64 vs 0.53). Optimal ensemble weight converges to ~90% Bayesian. This is expected: Elo + form features capture similar signal to the Bayesian posterior means, but the Bayesian model also learns attack/defence asymmetries from the full score distribution that are hard to capture with 5-game rolling features.

---

## D-12: Ensemble Weight Optimisation

**Decision:** Bayesian weight is optimised with `scipy.minimize_scalar` on the Brier score over the holdout set, bounded to [0.1, 0.9] to prevent degenerate one-model solutions.

**Reasoning:** Brier score is a proper scoring rule that directly measures calibration + resolution. Optimising it on held-out data is unbiased (no leakage). The [0.1, 0.9] bound ensures neither model is completely ignored — if one model degrades, the ensemble degrades gracefully rather than becoming a pure single-model.

---

## D-13: Isotonic Calibration Approach

**Decision:** Per-class isotonic regression fitted on ensemble outputs vs. holdout outcomes. Each class (H, D, A) has its own independent calibrator; outputs are renormalised to sum to 1.

**Reasoning:** Isotonic regression is non-parametric and handles multimodal calibration curves. Per-class fitting is simpler than multivariate approaches (e.g., Dirichlet calibration) and works well in practice for 3-class football prediction. Renormalisation preserves valid probability outputs after independent class calibration.

**Minimum sample requirement:** ≥30 holdout samples (enforced in code). With ~2,000 holdout matches, calibration is well-fitted.

---

## D-14: Bracket Simulation — Group Stage Approach

**Decision:** The full-tournament bracket simulation (`simulate_full_tournament`) simulates each group's remaining matches inside every Monte Carlo iteration. Each pair plays once (6 matches per group), all at neutral venue (`neutral=True`).

**Reasoning:** Simulating group matches within each bracket iteration correctly captures correlation between group results and knockout bracket position. This is essential for computing accurate conditional probabilities (e.g., "given Turkey wins the group, who do they meet in R16?").

**Contrast with group_sim.py:** The existing `group_sim.simulate_group` generates all ordered pairs (each pair plays twice, home and away) with `neutral=False`. This was the Phase 1 design and is kept for the Group D detailed view. The bracket sim uses the correct 6-match-per-group format.

---

## D-15: Bracket Seeding Approximation

**Decision:** The WC2026 R32 bracket is seeded into 4 sections of 8, with groups A–C in Section 1, D–F in Section 2, G–I in Section 3, J–L in Section 4. Group winners occupy higher-seed slots within each section.

**Reasoning:** FIFA had not published the exact R32 draw table at the model's knowledge cutoff (August 2025). The approximation above is consistent with the general WC bracket format. The seeding ensures teams from the same group cannot meet before the QF (correct by FIFA rule).

**When to update:** Once FIFA publishes the official bracket draw (typically ~6 months before the tournament), update `_seed_bracket()` to match exactly.

---

## D-16: Tournament Advance Probability — Two Numbers

**Decision:** Two advance probabilities are reported:
- `p_advance` from `group_sim` (Group D simulation) = P(top-2 in Group D) ≈ 12%
- `p_advance` from `bracket_sim` (full tournament simulation) = P(qualified by any route — top-2 OR best-3rd-place) ≈ 33%

**Reasoning:** The Group D sim is simpler and focuses on top-2 auto-qualification. The bracket sim additionally accounts for the 3rd-place route (8/12 third-place teams advance). The two numbers answer different questions and are shown in different UI sections.

---

## D-17: Bracket Probability Semantics

**Decision:** All bracket probabilities are ABSOLUTE (not conditional on qualifying):
- `p_advance` = P(qualified from group)
- `p_r16` = P(reached R16) — equivalent to won R32 match
- `p_qf` = P(reached QF)
- `p_sf`, `p_final`, `p_champion` similarly

**Reasoning:** Absolute probabilities are more intuitive to the end user ("what is Turkey's chance of winning the World Cup?") and are directly comparable across teams regardless of group difficulty. Conditional probabilities (e.g., "given they qualify, P(champion)") would be misleading when comparing a strong team vs. a weak team that qualifies 10% vs. 80% of the time.
