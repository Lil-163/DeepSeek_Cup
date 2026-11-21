# Literature Matrix — 2026 ShuWei Cup Problem C: State Estimation and Decision Optimization for Unmanned Supermarket Restocking

**Scope.** Core sources (12) for the three sub-problems: Q1 state/observation-reliability modeling from first-person 21-keypoint hand trajectories; Q2 prediction under missing observations and the decision of *when to request re-observation*; Q3 duration-distribution prediction, newsvendor-style reservation, and online scheduling of restocking tasks under release / precedence / budget / time-window constraints.

**DOI verification.** Every DOI below was verified on 2026-11-21 against the DOI registration databases — the Crossref REST API (`api.crossref.org`, for items 1, 3–12), the DataCite API (item 2) — plus the arXiv landing page (item 2) and publisher/library records where noted. Volume, issue, and page numbers are taken from these registry records, not from memory. No DOI or page range is fabricated.

**Theme coverage.**

| Theme | Sources |
|---|---|
| 1. State-space / Kalman filtering, hand & trajectory motion estimation | 1, 2 |
| 2. Intermittent observations, event-triggered (re-)sampling, POMDP, information value | 3, 4, 5, 6 |
| 3. Missing data and unreliable-observation statistics | 7 |
| 4. Duration / quantile prediction (quantile regression, gradient-boosted quantile loss) | 8, 9 |
| 5. Stochastic / single-machine / dynamic scheduling; newsvendor reservation | 10, 11, 12 |

---

## 1. Durbin & Koopman (2012) — State-space methods (Theme 1)

- **Full citation.** Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford: Oxford University Press.
- **DOI.** [10.1093/acprof:oso/9780199641178.001.0001](https://doi.org/10.1093/acprof:oso/9780199641178.001.0001) — verified via Crossref (publisher: Oxford University Press; print ISBN 9780199641178; "Second Edition"; issued 2012).
- **Research question.** How to model, filter, smooth, forecast, and estimate parameters of time series in a unified linear Gaussian state-space framework, including series with missing observations.
- **Method & assumptions.** Linear Gaussian state-space models with observation and state disturbances; Kalman filter and state/disturbance smoothers; exact diffuse initialization; maximum-likelihood estimation of hyperparameters; dedicated treatment of missing observations (skipping the update step while keeping the likelihood valid); multivariate and nonstationary extensions. Assumes linear dynamics and Gaussian noise.
- **Evaluation metrics.** Log-likelihood, one-step-ahead prediction errors and their variances, model diagnostics, information criteria for model selection.
- **Transfer to Problem C.**
  - **Q1:** Model each hand's 21-keypoint 3D trajectory (or a reduced state such as per-keypoint position/velocity) as a state-space model; treat bad frames (missing / occluded / near-edge / quality-flagged) as missing observations or as observations with inflated noise variance derived from the quality flag; the Kalman smoother returns posterior state estimates **with covariance**, i.e., a principled, per-frame observation-reliability and state-uncertainty metric — exactly what Q1 asks for. Semantic segment labels can switch model parameters (multi-regime / piecewise models).
  - **Q2:** The filter recursion is the workhorse for predicting state and covariance *during* occlusion gaps (propagation without update); the propagated covariance is the natural input to a "request re-observation" rule.
- **Caveats for this problem.** Linearity and Gaussianity may be violated by fast occlusion jumps and by the weakly calibrated depth coordinate of keypoint detectors; 21 keypoints × 2 hands × 30 fps calls for factored or per-landmark models to stay computationally light; the book gives the estimation machinery, not a decision policy (pair with sources 4–6).

## 2. Zhang et al. (2020) — MediaPipe Hands (Theme 1)

- **Full citation.** Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C.-L., & Grundmann, M. (2020). MediaPipe Hands: On-device Real-time Hand Tracking. *arXiv preprint arXiv:2006.10214*; presented at the CVPR 2020 Workshop on Computer Vision for Augmented and Virtual Reality, Seattle, WA, USA.
- **DOI.** [10.48550/arXiv.2006.10214](https://doi.org/10.48550/arXiv.2006.10214) — verified via DataCite (arXiv-issued DOI; creators, title, year 2020 confirmed) and the arXiv abstract page.
- **Research question.** Real-time, on-device hand tracking that predicts a hand skeleton (2.5D landmarks) from a single RGB camera, robust to partial visibility and self-occlusion.
- **Method & assumptions.** Two-stage pipeline: (1) a single-shot palm detector (encoder-decoder, focal loss) that produces an oriented hand bounding box; (2) a hand landmark model that regresses **21 landmarks with x, y and wrist-relative depth** plus a hand-presence confidence and a left/right handedness head. Trained on in-the-wild (6K), in-house gesture (10K), and synthetic (100K) images; 2D coordinates learned from real + synthetic data, depth learned only from synthetic data. During tracking the detector is re-triggered only when the hand-presence confidence drops below a threshold; otherwise the crop is propagated from the previous frame's landmarks.
- **Evaluation metrics.** Palm detector average precision: 95.7% (decoder + focal loss); landmark regression MSE normalized by palm size: 13.4% (real + synthetic training); on-device latency (e.g., 16.1 ms on Pixel 3 for the "Full" model).
- **Transfer to Problem C.**
  - **Q1:** The competition's data format (21 3D keypoints per hand, 30 fps) matches this landmark topology; the paper explains the *noise structure* of such data — z is wrist-relative depth learned only from synthetic images (weakly calibrated, no physical units), x/y are normalized by the palm crop (scale-dependent, heteroscedastic, degraded near frame edges) — which is exactly the regime of the "bad frame / near-edge / occlusion" quality markers. The hand-presence confidence that re-triggers detection is a real-world template for the competition's quality flags and for deciding when tracking must be re-acquired.
  - **Q2:** The detection-re-trigger mechanism is an existence proof of confidence-driven re-observation; the paper's per-frame independence (no temporal model) is precisely the gap that Q1/Q2's state-space prediction fills.
- **Caveats for this problem.** 2.5D, not metric 3D — the depth coordinate cannot be treated as a physical measurement without rescaling; workshop paper / preprint (not a peer-reviewed journal); no per-landmark uncertainty is output (must be supplied by a Kalman layer or quantile machinery); performance degrades under heavy occlusion, which is the competition's core failure mode rather than something the paper solves.

## 3. Sinopoli et al. (2004) — Kalman filtering with intermittent observations (Theme 2)

- **Full citation.** Sinopoli, B., Schenato, L., Franceschetti, M., Poolla, K., Jordan, M. I., & Sastry, S. S. (2004). Kalman filtering with intermittent observations. *IEEE Transactions on Automatic Control*, 49(9), 1453–1464.
- **DOI.** [10.1109/TAC.2004.834121](https://doi.org/10.1109/TAC.2004.834121) — verified via Crossref (IEEE TAC, vol. 49, issue 9, pp. 1453–1464, 2004).
- **Research question.** How does random loss of observations affect Kalman-filter estimation quality, and under what arrival probability does the expected estimation error stay bounded?
- **Method & assumptions.** Standard linear-Gaussian state-space model whose measurements are dropped i.i.d. (Bernoulli arrival probability λ); the filter runs a random Riccati recursion; the authors derive upper and lower bounds on the expected error covariance E[P_k] and prove the existence of a **critical arrival probability λ_c**: E[P_k] is bounded if and only if λ > λ_c (for a scalar system with transition a, λ_c = 1 − 1/a²; in general λ_c depends on the eigenvalues of A and the structure of C). Assumptions: i.i.d. Bernoulli loss, known model, ideal filter computation.
- **Evaluation metrics.** Trace of the expected estimation-error covariance; the critical probability; Monte-Carlo comparison of bounds.
- **Transfer to Problem C.**
  - **Q2:** Occlusion ⇒ intermittent observations. The paper quantifies the *phase transition*: during a gap the expected covariance grows geometrically, and below the critical arrival rate prediction-only estimation diverges — this gives a rigorous criterion for when pure prediction becomes worthless and a re-observation must be requested. The modified Riccati recursion is exactly the computation needed to forecast "how uncertain will we be in t frames if we do nothing", which drives the request decision.
  - **Q1:** The same recursion, with per-frame quality-dependent observation noise, upgrades the plain Kalman filter into a reliability-aware estimator.
- **Caveats for this problem.** i.i.d. Bernoulli loss — real occlusions are bursty and state-dependent (MNAR in the sense of source 7), so treat λ as time-varying / conditioned on context; no observation cost, latency, or decision policy (only estimation analysis); assumes known linear model parameters.

## 4. Trimpe & D'Andrea (2014) — Event-based estimation with variance-based triggering (Theme 2)

- **Full citation.** Trimpe, S., & D'Andrea, R. (2014). Event-based state estimation with variance-based triggering. *IEEE Transactions on Automatic Control*, 59(12), 3266–3281.
- **DOI.** [10.1109/TAC.2014.2351951](https://doi.org/10.1109/TAC.2014.2351951) — verified via Crossref (IEEE TAC, vol. 59, issue 12, pp. 3266–3281, 2014).
- **Research question.** When should a sensor send a new measurement so that the estimation error stays within a required bound while communication is minimized?
- **Method & assumptions.** The sensor and the receiver run identical estimators; the sensor transmits a measurement **only when the predicted variance of the estimate of a designated output exceeds a threshold δ** (variance-based triggering). Because the triggering law is known on both sides, the receiver can compute the exact error covariance of the shared estimate and the authors provide guarantees bounding the covariance while quantifying the resulting average communication rate. Assumes an ideal channel, a known linear system, and noiseless triggering logic.
- **Evaluation metrics.** Guaranteed estimation-error bounds vs. average communication (re-observation) rate; simulation/experimental validation.
- **Transfer to Problem C.**
  - **Q2:** This is the direct template for *adaptive re-observation*: keep propagating state and covariance through the occlusion; issue a re-observation request exactly when the predicted variance of the relevant hand state exceeds the threshold set by the error-cost budget. Fixed-interval re-checking is the degenerate special case (time-triggered); the variance trigger is the adaptive policy the problem asks to compare. The framework explicitly handles the delay between request and fresh observation, matching the "return delay" element.
  - **Q1:** Provides the covariance-propagation logic that turns the Kalman filter into a *scheduled-observation* estimator.
- **Caveats for this problem.** The threshold δ must come from a cost model, which the paper does not provide — combine with sources 6 (information value) and 5 (POMDP rewards); triggering is on a scalar output, so one must choose which keypoint / state combination to monitor; single-sensor, single-channel setup (two hands / multiple cameras need extensions); no monetary fee structure built in.

## 5. Kaelbling, Littman & Cassandra (1998) — POMDP planning (Theme 2)

- **Full citation.** Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and acting in partially observable stochastic domains. *Artificial Intelligence*, 101(1–2), 99–134.
- **DOI.** [10.1016/S0004-3702(98)00023-X](https://doi.org/10.1016/S0004-3702(98)00023-X) — verified via Crossref (Artificial Intelligence, vol. 101, issue 1–2, pp. 99–134, 1998).
- **Research question.** How can an agent plan optimally when it observes only noisy, partial information about an underlying state?
- **Method & assumptions.** POMDP = (S, A, T, Z, O, R, γ); the belief state is a sufficient statistic, reducing planning to a belief-state MDP; exact value iteration over the continuous belief simplex (Sondik's results; the witness algorithm); complexity results (finite-horizon POMDPs are PSPACE-hard); survey of approximations (grid-based, policy search, QMDP-style shortcuts). Assumes known transition/observation/reward models.
- **Evaluation metrics.** Expected discounted reward; policy quality vs. computational cost; belief-space size.
- **Transfer to Problem C.**
  - **Q2:** The "when to request re-observation" decision is a POMDP action: states = true hand state / task progress; actions = {keep predicting, request re-observation}; observations = keypoint frames with quality flags; rewards = −estimation error − observation cost − latency. The POMDP belief update over intermittent observations is the general form of source 3's covariance recursion (the Gaussian special case). The optimal POMDP policy *is* the optimal adaptive re-check schedule; fixed/periodic re-checking is a restricted policy against which it benchmarks.
  - **Q3:** Online dispatching with predicted (uncertain) durations can be framed as a belief-space MDP over remaining durations; EDD-type rules are heuristic policies in this frame.
- **Caveats for this problem.** Exact methods are intractable for 42-D continuous keypoint states — one must compress the belief (Gaussian summaries, task-level features) or use point-based/approximate solvers; requires observation and transition models that Q1 must estimate from data; provides the decision framework but no sensing-noise model of its own.

## 6. Howard (1966) — Information value theory (Theme 2)

- **Full citation.** Howard, R. A. (1966). Information value theory. *IEEE Transactions on Systems Science and Cybernetics*, 2(1), 22–26.
- **DOI.** [10.1109/TSSC.1966.300074](https://doi.org/10.1109/TSSC.1966.300074) — verified via Crossref (IEEE Trans. Systems Science and Cybernetics, vol. 2, issue 1, pp. 22–26, 1966).
- **Research question.** What is the maximum price a rational decision maker should pay for information in a decision problem under uncertainty?
- **Method & assumptions.** Decision-theoretic framework: value of information = expected payoff of the optimal decision *with* the information minus that *without* it; clairvoyance (perfect information) yields the expected value of perfect information (EVPI); the "wizard" model (imperfect, noisy information) yields the expected value of imperfect information (EVII); Bayesian revision of probabilities on receiving information; linear pricing rule for information. Assumes known priors, likelihoods, and payoffs.
- **Evaluation metrics.** EVPI / EVII in expected monetary value; sensitivity to priors and payoff structure.
- **Transfer to Problem C.**
  - **Q2:** A re-observation request is a purchase of imperfect information: request it iff its EVII exceeds the re-observation cost (fee + expected latency damage to estimation/scheduling). EVPI bounds the total budget it is ever rational to spend on re-observation.
  - **Q3:** The paid re-check of a duration forecast is the same calculus — expected scheduling-cost reduction from a sharper duration distribution vs. the re-check fee. This supplies the cost side that sources 3–5 lack and lets one convert covariance thresholds (source 4) into money.
- **Caveats for this problem.** Single-decision, static framing — sequential/multi-stage re-observation needs the POMDP generalization (source 5); assumes probabilities and payoffs are known (must be estimated from Q1/Q3 models); does not specify how to obtain posteriors from keypoint observations.

## 7. Little & Rubin (2019) — Statistical analysis with missing data (Theme 3)

- **Full citation.** Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed., Wiley Series in Probability and Statistics). Hoboken, NJ: John Wiley & Sons.
- **DOI.** [10.1002/9781119482260](https://doi.org/10.1002/9781119482260) — verified via Crossref (Wiley monograph; "Statistical Analysis with Missing Data, Third Edition"; issued 2019-04-12; print ISBN 9780470526798).
- **Research question.** How should inference proceed when part of the data is unobserved: when is ignoring the missingness safe, and what corrections are needed otherwise?
- **Method & assumptions.** Missingness taxonomy — MCAR / MAR / MNAR — and the concept of ignorability, originating in Rubin (1976), *Inference and missing data*, Biometrika, 63(3), 581–592, doi:[10.1093/biomet/63.3.581](https://doi.org/10.1093/biomet/63.3.581) (verified via Crossref); likelihood-based inference (factored likelihood, EM); multiple imputation; Bayesian methods; pattern-mixture and selection models for MNAR data; sensitivity analysis. Key assumption: validity of likelihood-based analysis relies on MAR (or MCAR) plus parameter distinctness.
- **Evaluation metrics.** Bias, efficiency, and coverage of intervals; imputation diagnostics; sensitivity of conclusions to MNAR departures.
- **Transfer to Problem C.**
  - **Q1:** Quality flags / bad frames are missingness indicators. Before smoothing or imputing, classify the mechanism: near-edge or quality-flagged loss is plausibly MAR (predictable from visible covariates), while occlusion by the shelf/object is likely MNAR (missingness depends on the unobserved hand position). Kalman filtering that skips missing frames is exactly likelihood-based inference under MAR; under suspected MNAR, add selection-model terms or sensitivity analysis. This formalizes the "observation reliability" story of Q1.
  - **Q2:** Prediction during occlusion uses the MAR machinery; the MNAR diagnosis warns when a pure filter will drift — strengthening the case for paid re-observation.
  - **Q3:** Missing prefix features in duration models are missing covariates; imputation vs. likelihood handling applies to training data.
- **Caveats for this problem.** General-purpose statistics with no pose-specific missingness models; multiple imputation is batch-oriented — at 30 fps one should use sequential filtering/EM rather than multiply-impute; MAR is an assumption that must be argued from the quality-flag metadata, not taken for granted.

## 8. Koenker & Hallock (2001) — Quantile regression (Theme 4)

- **Full citation.** Koenker, R., & Hallock, K. F. (2001). Quantile regression. *Journal of Economic Perspectives*, 15(4), 143–156.
- **DOI.** [10.1257/jep.15.4.143](https://doi.org/10.1257/jep.15.4.143) — verified via Crossref (JEP, vol. 15, issue 4, pp. 143–156, 2001).
- **Research question.** What does quantile regression estimate, and why is modeling conditional quantiles — rather than the conditional mean — the right tool under asymmetric loss, heteroscedasticity, and risk assessment?
- **Method & assumptions.** Conditional quantile functions estimated by minimizing an asymmetrically weighted sum of absolute residuals (the "check" function ρ_τ); the median (τ = 0.5) minimizes absolute error; equivariance properties; linear-programming formulation for computation; applications to CEO pay, food expenditure, infant birthweight, and value-at-risk. Assumes a parametric (typically linear) quantile surface and i.i.d. data.
- **Evaluation metrics.** Quantile-specific coefficient estimates and confidence intervals; quantile-process plots; coverage.
- **Transfer to Problem C.**
  - **Q3:** Predict conditional duration quantiles from prefix features (0.2 / 0.6 / 1.0 s reveal levels) per restocking task; the newsvendor-optimal reservation is *exactly* a quantile at the critical fractile c_u/(c_u + c_o) (source 12), so quantile regression is the estimation engine of the reservation decision. Asymmetric over- vs. under-reservation costs map directly onto ρ_τ weights. Fitting several τ values per reveal level exposes how uncertainty shrinks with information — the quantitative "value of the graded information reveal".
  - **Q2/Q1:** Secondary use: conditional quantiles of trajectory noise per quality class can calibrate observation-error models.
- **Caveats for this problem.** Linear-in-parameters baseline is likely too rigid for keypoint-derived features — upgrade to gradient-boosted quantile models (source 9); independently fitted quantiles can cross (needs monotone post-processing); extreme-tail quantiles (τ → 0, 1) are poorly estimated with small samples.

## 9. Chen & Guestrin (2016) — XGBoost (Theme 4)

- **Full citation.** Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (KDD '16) (pp. 785–794). New York, NY: ACM.
- **DOI.** [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785) — verified via Crossref (ACM, KDD '16 proceedings, pp. 785–794, 2016; subtitle "A Scalable Tree Boosting System").
- **Research question.** How to make gradient tree boosting scalable, regularized, and flexible enough for arbitrary differentiable objectives on large datasets?
- **Method & assumptions.** Second-order (Newton-style) gradient boosting; regularization (L1/L2, shrinkage, subsampling) in the objective; approximate split finding via a weighted quantile sketch; sparsity-aware split handling; out-of-core and cache-aware computation; custom objectives — which includes the asymmetric pinball (quantile) loss optimized via subgradients, yielding gradient-boosted quantile regression. Assumes i.i.d. tabular training data with engineered features.
- **Evaluation metrics.** Accuracy/AUC and wall-clock time on public benchmarks (Higgs, Allstate, click prediction); model-size and speed tradeoffs.
- **Transfer to Problem C.**
  - **Q3:** Train per-quantile gradient-boosted models (pinball loss) on prefix summary features (keypoint statistics, motion features, semantic segment) for each reveal level (0.2 / 0.6 / 1.0 s) → a full predicted duration distribution per task; the predicted quantiles feed the newsvendor reservation (source 12) and the predicted variance feeds the re-check EVII (source 6); feature-importance output identifies which prefix statistics carry duration information.
  - **Q2:** Predicted state/distribution variance from such models can serve as the trigger statistic for re-observation requests.
- **Caveats for this problem.** Black-box: requires engineered prefix features (raw 21-keypoint trajectories must be summarized); pinball loss is non-differentiable at zero (handled by subgradients — standard practice); tail quantiles are unreliable with small data; no built-in uncertainty beyond the fitted quantiles (conformal prediction could wrap it, optional); hyperparameter tuning cost.

## 10. Pinedo (2016) — Scheduling: theory, algorithms, and systems (Theme 5)

- **Full citation.** Pinedo, M. L. (2016). *Scheduling: Theory, Algorithms, and Systems* (5th ed., 670 pp.). Cham: Springer International Publishing.
- **DOI.** [10.1007/978-3-319-26580-3](https://doi.org/10.1007/978-3-319-26580-3) — verified via Crossref (Springer book, 2016; ISBN 9783319265780 / 9783319265803); 5th edition and page extent confirmed via Scopus and library catalog records.
- **Research question.** Systematic treatment of scheduling models: which schedules are optimal, which algorithms find them, and which problems are hard — for deterministic and stochastic settings.
- **Method & assumptions.** Single- and multi-machine models; classical optimality results for single machines (EDD minimizes maximum lateness L_max; WSPT minimizes Σw_jC_j; SPT minimizes ΣC_j); release dates, precedence constraints, due dates, time windows; dynamic programming and branch-and-bound; dispatching rules; online scheduling with competitive analysis; a chapter on stochastic scheduling with random processing times and expected-objective policies. Assumes processing times are known (deterministic) or fully specified by distributions.
- **Evaluation metrics.** Optimality of objective values (makespan, ΣC_j, L_max, Σw_jC_j); worst-case competitive ratios; expected objective values; computational complexity.
- **Transfer to Problem C.**
  - **Q3:** The single restocking device is a single machine: tasks = jobs, durations = processing times, 释放 constraints = release dates, 前置 constraints = precedence, time windows = due-date/deadline constraints, budget = a knapsack-style selection constraint. EDD is the required online baseline and is justified by the book's L_max result; WSPT/SPT variants give alternative baselines; the stochastic-scheduling chapter supplies expected-lateness evaluation of reservation policies; DP/knapsack formulations handle the budget constraint.
- **Caveats for this problem.** Classical models assume known or fully specified durations — here they are predicted quantile distributions conditioned on prefixes (sources 8–9); the book is mostly offline — dynamic re-optimization needs source 11; no observation/re-observation cost channel.

## 11. Ouelhadj & Petrovic (2009) — Survey of dynamic scheduling (Theme 5)

- **Full citation.** Ouelhadj, D., & Petrovic, S. (2009). A survey of dynamic scheduling in manufacturing systems. *Journal of Scheduling*, 12(4), 417–431.
- **DOI.** [10.1007/s10951-008-0090-8](https://doi.org/10.1007/s10951-008-0090-8) — verified via Crossref (Journal of Scheduling, vol. 12, issue 4, pp. 417–431; published online 2008, issue 2009).
- **Research question.** What methods exist for scheduling in dynamic, uncertain environments (arriving jobs, disruptions, uncertain processing times), and how should reactive behavior be organized?
- **Method & assumptions.** Taxonomy of approaches: completely reactive vs. predictive-reactive vs. robust pro-active scheduling; rescheduling policies — periodic, event-driven, hybrid; rescheduling methods — right-shift, partial rescheduling, full regeneration; efficiency vs. stability tradeoff; rolling-horizon techniques; metaheuristic and multi-agent implementations. Assumes disruptions and processing times are observable when they occur.
- **Evaluation metrics.** Classical objectives (makespan, tardiness) jointly with stability measures (number of shifted jobs, deviation from the baseline schedule) and computational effort.
- **Transfer to Problem C.**
  - **Q3:** Restocking is inherently dynamic (durations revealed progressively, bad frames, overruns): adopt predictive-reactive scheduling — maintain a baseline, reschedule on events (re-check outcomes, deadline overruns). The event-driven rescheduling policy is the scheduler-side hook for Q2's "request re-observation" decisions; right-shift repair handles overruns; stability metrics matter because re-optimization churn consumes the re-observation budget; rolling-horizon re-planning consumes updated duration quantiles as prefixes lengthen.
- **Caveats for this problem.** A survey — no single ready-to-run algorithm; manufacturing terminology (machines, setup times) must be mapped to restocking; assumes durations are observed online, whereas here only prefix-based predictions exist — so the trigger logic must use predicted variance (source 4) rather than observed deviation alone; stability vs. efficiency weighting must be tuned to the competition's objective.

## 12. Qin, Wang, Vakharia, Chen & Seref (2011) — Newsvendor review (Theme 5)

- **Full citation.** Qin, Y., Wang, R., Vakharia, A. J., Chen, Y., & Seref, M. M. H. (2011). The newsvendor problem: Review and directions for future research. *European Journal of Operational Research*, 213(2), 361–374.
- **DOI.** [10.1016/j.ejor.2010.11.024](https://doi.org/10.1016/j.ejor.2010.11.024) — verified via Crossref (EJOR, vol. 213, issue 2, pp. 361–374, 2011).
- **Research question.** Synthesis of the single-period (newsvendor) inventory problem: the basic model, its extensions, and open research directions.
- **Method & assumptions.** Basic model: choose a quantity q to maximize expected profit under stochastic demand; optimality: q* = F⁻¹(c_u/(c_u + c_o)) — the critical-fractile quantile of the demand distribution. Extensions: price-dependent demand, risk-averse objectives (CVaR), bounded rationality, demand-distribution estimation, multi-item settings with constraints, supply-chain coordination. Assumes a known or estimable demand distribution and known unit overage/underage costs.
- **Evaluation metrics.** Expected profit; sensitivity to the cost ratio; the profit gain from improved demand forecasts.
- **Transfer to Problem C.**
  - **Q3:** The reservation-duration decision is a newsvendor quantity: c_u = cost of under-reserving (task exceeds reservation → deadline miss / cascade / penalty), c_o = cost of over-reserving (idle device time / budget waste); reserve the critical-fractile quantile of the prefix-conditioned duration distribution. Sharper distributions from longer reveals reduce expected cost — quantifying the value of the graded information reveal; the multi-item / constrained extensions map to many tasks under a shared budget; risk-averse variants (CVaR) apply if the objective penalizes tail lateness.
- **Caveats for this problem.** The duration distribution is not given — it must come from the quantile models of sources 8–9 conditioned on the visible prefix; single-period myopia ignores schedule coupling (a reservation affects downstream tasks), so embed newsvendor reservations inside rolling re-optimization (sources 10–11); no observation-cost channel — pair with source 6 for paid re-checks.

---

## Summary Mapping Table

| # | Source (short) | Applies to | One-sentence transfer point |
|---|---|---|---|
| 1 | Durbin & Koopman 2012 | Q1 (Q2) | Kalman filtering/smoothing over the 21-keypoint trajectory treats bad frames as missing and outputs posterior state **plus covariance** = Q1's observation-reliability metric. |
| 2 | Zhang et al. 2020 (MediaPipe Hands) | Q1 (Q2) | Explains the data modality's noise structure (wrist-relative depth, palm-normalized x/y, presence confidence) and why bad-frame flags and re-acquisition exist. |
| 3 | Sinopoli et al. 2004 | Q2 (Q1) | Intermittent-observation Kalman theory gives the critical arrival rate and covariance recursion that quantify when prediction during occlusion diverges. |
| 4 | Trimpe & D'Andrea 2014 | Q2 (Q1) | Variance-based triggering is the canonical adaptive rule: request re-observation when predicted state variance exceeds the error-cost threshold. |
| 5 | Kaelbling et al. 1998 | Q2 (Q3) | POMDP belief-space planning yields the optimal adaptive re-check policy trading estimation error against observation cost and latency. |
| 6 | Howard 1966 | Q2, Q3 | EVPI/EVII tell exactly how much a re-observation or duration re-check is worth before paying for it. |
| 7 | Little & Rubin 2019 | Q1 (Q2, Q3) | MCAR/MAR/MNAR classification of bad frames/occlusion determines whether filtering-out missing observations is valid or needs MNAR correction. |
| 8 | Koenker & Hallock 2001 | Q3 | Conditional quantile regression directly estimates the duration quantiles that newsvendor reservation requires. |
| 9 | Chen & Guestrin 2016 | Q3 (Q2) | Gradient-boosted pinball-loss models turn prefix features into full duration distributions per reveal level (0.2/0.6/1.0 s). |
| 10 | Pinedo 2016 | Q3 | Single-machine theory (EDD for L_max, WSPT/SPT, release/precedence/time-window models) supplies the required online scheduling baselines and structure. |
| 11 | Ouelhadj & Petrovic 2009 | Q3 (Q2) | Predictive-reactive / event-driven rescheduling converts updated duration forecasts and re-check events into schedule revisions with stability control. |
| 12 | Qin et al. 2011 | Q3 | Newsvendor critical-fractile logic converts the predicted duration distribution into an optimal reservation under asymmetric over/under-reservation costs. |

