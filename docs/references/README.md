# References

Primary sources underpinning the methodology in `docs/leakage_audit.md` and
`src/deep_lob/audit.py`. Each entry includes a DOI or other stable
publisher/preprint link, verified independently for this repository (not
guessed) at the time of writing; if a link later rots, the title/author/
venue metadata is sufficient to relocate the source.

## Overlapping windows, cross-validation, and leakage in time series

- Bergmeir, C., & Benítez, J. M. (2012). "On the use of cross-validation for
  time series predictor evaluation." *Information Sciences*, 191, 192–213.
  DOI: [10.1016/j.ins.2011.12.028](https://doi.org/10.1016/j.ins.2011.12.028)
  — Foundational argument for why standard (random) k-fold cross-validation
  is invalid for temporally dependent / autocorrelated data.

- Cerqueira, V., Torgo, L., & Mozetič, I. (2020). "Evaluating time series
  forecasting models: An empirical study on performance estimation
  methods." *Machine Learning*, 109(11), 1997–2028.
  DOI: [10.1007/s10994-020-05910-7](https://doi.org/10.1007/s10994-020-05910-7)
  ([arXiv:1905.11744](https://arxiv.org/abs/1905.11744)) — Empirical
  comparison of cross-validation vs. out-of-sample evaluation under
  temporal dependence and overlapping samples.

- López de Prado, M. (2018). *Advances in Financial Machine Learning*.
  Wiley. ISBN: [978-1-119-48208-6](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086).
  Chapter 7, "Cross-Validation in Finance." — Introduces purged and
  embargoed k-fold cross-validation specifically to prevent label/feature
  overlap leakage between train and test folds in financial time series.

- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). "The
  Probability of Backtest Overfitting." *Journal of Computational Finance*,
  20(4), 39–69. DOI: [10.21314/JCF.2016.322](https://doi.org/10.21314/JCF.2016.322)
  (preprint: [SSRN 2326253](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253))
  — Formalizes how repeated in-sample optimization on overlapping/finite
  data inflates apparent (but non-generalizing) skill. Note: journal issue
  date is 2017, cited as (2014) in some secondary sources after the
  original working-paper year; this ledger uses the journal's own
  publication year.

## Limit order book modeling

- Zhang, Z., Zohren, S., & Roberts, S. (2019). "DeepLOB: Deep Convolutional
  Neural Networks for Limit Order Books." *IEEE Transactions on Signal
  Processing*, 67(11), 3001–3012.
  DOI: [10.1109/TSP.2019.2907260](https://doi.org/10.1109/TSP.2019.2907260)
  ([arXiv:1808.03668](https://arxiv.org/abs/1808.03668)) — Source
  architecture referenced by `src/deep_lob/models.py::DeepLOBModel`; see
  also `docs/deeplob_paper.tex` in this repository.

- Cont, R., Stoikov, S., & Talreja, R. (2010). "A Stochastic Model for
  Order Book Dynamics." *Operations Research*, 58(3), 549–563.
  DOI: [10.1287/opre.1090.0780](https://doi.org/10.1287/opre.1090.0780)
  — Baseline reference for order book queueing/arrival dynamics; useful
  context for how far the synthetic generator in `simulator.py` is from a
  realistic microstructure model (it implements neither queueing nor
  order-flow dependence).

- Sirignano, J., & Cont, R. (2019). "Universal features of price
  formation in financial markets: perspectives from deep learning."
  *Quantitative Finance*, 19(9), 1449–1459.
  DOI: [10.1080/14697688.2019.1622295](https://doi.org/10.1080/14697688.2019.1622295)
  ([arXiv:1803.06917](https://arxiv.org/abs/1803.06917)) — Discusses what
  real order-book-driven predictive features look like, as a contrast
  point for interpreting the null result on this repository's synthetic
  data.

## Scope note

None of the above are cited as endorsing any specific accuracy, PnL, or
Sharpe figure produced by this repository. They are cited for
*methodology* (how to avoid leaking test information into train sets when
evaluating on temporally dependent, overlapping-window data).
