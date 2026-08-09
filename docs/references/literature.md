# DeepLOB Literature Review: Data Leakage and Time-Series Evaluation

## 1. Data Leakage in ML Generally
**Citation:** Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in ML-based science. *Patterns*, 4(9), 100804.
**Link:** https://arxiv.org/abs/2207.07048
**Claims:** The authors systematically survey 329 papers across 17 scientific fields, finding that data leakage is a pervasive cause of reproducibility failures. They construct a detailed taxonomy of eight distinct types of leakage and demonstrate that when leakage is corrected, complex ML models frequently fail to outperform simple baselines like logistic regression.
**Relation:** Precedent — establishes that our finding of artificially inflated accuracy is part of a systemic cross-disciplinary crisis, not just an isolated anomaly.

## 2. Leakage in Time-Series / Financial ML
**Citation:** López de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.
**Link:** https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086
**Claims:** The author argues that standard k-fold cross-validation fails catastrophically on financial time-series because forward-looking labels naturally overlap in time, allowing the model to peek into the future. To prevent this serial correlation from leaking information, the book introduces "purging" (removing training data that overlaps with test labels) and "embargoing" (adding a buffer period immediately following the test set).
**Relation:** Method we use — provides the exact theoretical framework and corrective methodology (purged/embargoed splits) that we employ to drop our model's performance back to the baseline.

## 3. Backtest Overfitting and Inflated Performance
**Citation:** Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014). Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance. *Notices of the American Mathematical Society*, 61(5), 458-471.
**Link:** https://doi.org/10.1090/noti1105
**Claims:** This paper demonstrates mathematically that running a high number of strategy configurations on historical data virtually guarantees finding an apparently highly profitable strategy, even when the underlying data is completely random. The authors argue that unreported multiple-testing ("winner-picking") acts as a form of overfitting that ensures out-of-sample failure.
**Relation:** Precedent — mathematically demonstrates how random noise can yield high performance metrics under flawed evaluation regimes, supporting our empirical results on null-data.

## 4. Reproducibility and Seed Sensitivity in ML
**Citation:** Picard, D. (2021). Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision. *arXiv preprint*.
**Link:** https://arxiv.org/abs/2109.08203
**Claims:** The author shows that merely changing the random seed for weight initialization and data shuffling can drastically alter a deep learning model's performance on standard benchmarks, even with all hyperparameters fixed. The paper highlights that single-seed reporting is fragile and that some "state-of-the-art" claims may simply be the result of stumbling onto a lucky seed.
**Relation:** Precedent — establishes a baseline for how much performance can vary by chance, throwing into sharp relief that our massive 37% accuracy gap is a structural evaluation failure, not mere seed variance.

## 5. Negative-Control / Null-Data Methodology
**Citation:** *Could not verify.* 
**Link:** N/A
**Claims:** N/A
**Relation:** Gap — while papers like Bailey et al. (2014) use random data to demonstrate multiple-testing overfitting, I could not verify any existing work that explicitly uses a pure white-noise/null-data negative control *specifically to demonstrate the mechanical inflation caused by overlapping time-series cross-validation windows*.

## 6. Existing Tooling
**Citation:** Talagala, T. S. (2024). tsdataleaks: An R Package to Detect Potential Data Leaks in Forecasting Competitions. *arXiv preprint*.
**Link:** https://arxiv.org/abs/2402.10522
**Claims:** This paper introduces an R package explicitly designed to programmatically detect data leakage in time-series datasets, such as repeated patterns, scale-shifts, and concatenated blocks. It provides algorithms and visualizations to flag when test set information has inadvertently contaminated the training pool.
**Relation:** Precedent — demonstrates that the community is actively trying to build programmatic tripwires for time-series leakage, though it focuses on dataset construction flaws rather than structural evaluation logic flaws.

---

## Conclusion: What is Genuinely New?
To be blunt, the underlying mechanics of our finding are not new. The danger of overlapping windows in time-series is a known quantity (López de Prado, 2018), as is the fact that machine learning models can find spurious patterns in random noise (Bailey et al., 2014).

What is genuinely **new** about our result is the *direct, empirical isolation of the evaluation mechanic using a negative control*. 

When researchers demonstrate that purged cross-validation lowers model performance on real financial data, critics can argue that purging is too aggressive and destroys legitimate, short-term signal. By evaluating our pipeline on data with *provably zero signal*, we remove that defense completely. We isolate the overlapping split mechanic as the *sole* driver of the 71% vs 34% accuracy gap.

Our contribution is not discovering leakage; it is providing a clean, undeniable "smoking gun." It quantifies exactly how much accuracy standard random splitting hallucinates out of thin air, serving as a definitive pedagogical proof that ends the debate over whether time-series purging is strictly necessary.
