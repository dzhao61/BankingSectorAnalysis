# Concentration of the Australian Equity Market in 5 Major Banks

An information-theoretic analysis of stock price dependencies among Australia's five major banks using Mutual Information (MI) and Conditional Mutual Information (CMI).

---

## Motivation

Australia's equity market exhibits significant concentration in the banking sector, where five stocks account for roughly **25–30% of the ASX200 index** (S&P Global, 2025).

| Bank | ASX Ticker | ~ASX200 Weight |
|---|---|---|
| Commonwealth Bank | CBA | ~11% |
| National Australia Bank | NAB | ~5% |
| Westpac | WBC | ~5% |
| ANZ | ANZ | ~4% |
| Macquarie Group | MQG | ~3% |

If share price dependence between these banks is strong, it disproportionately drives ASX200 returns, volatility, and investor sentiment — reducing the diversification benefit of passive index investing. Superannuation funds and ETFs tracking the ASX200 inadvertently concentrate household wealth in this single sector.

---

## Research Question

> **To what extent do dependencies exist among the stock price movements of Australia's major banks, and do these dependencies persist after conditioning on broader stock market and macroeconomic influences?**

This project investigates the question using information-theoretic tools — treating MI and CMI as a form of *non-linear correlation*.

| Outcome | Interpretation |
|---|---|
| Low MI, Low CMI | Independent price dynamics (best case) |
| High MI, Low CMI | Market/macro-driven interdependence (systematic risk) |
| High MI, High CMI | Sector-specific interdependencies (idiosyncratic risk, worst case) |

---

## Method

Mutual and conditional mutual information are calculated over a 10-year period using the **KSG1 estimator** on **150-day rolling windows**, capturing time-varying dependencies among the banks' daily log returns.

**Pipeline:**
1. 10+ years of historical daily prices retrieved via the YFinance Python API
2. Daily log returns calculated and standardised; ACF performed
3. 150-day rolling window applied to capture time-varying dependencies
4. MI and CMI calculated between all bank pairs using the KSG1 estimator

**Conditioning Variables:**

| Factor | Proxy | Rationale |
|---|---|---|
| Broader stock market movement | Solactive ASX200 Ex-Top 20 Index (`DE000SL0CK81.SG`) | Controls for overall market trends while excluding the major banks. Best available free proxy. |
| Macroeconomic factors | Solactive Australia Government Bonds Index (`DE000SLA6QZ3.SG`) | Bond yields reflect expectations of interest rates, inflation, and other macro factors. |

**Additional Notes:**
- **Robustness checks:** Window size tested across 50–300 days; k-neighbours tested across 2–10. Final parameters set to window = 150 days, k = 4, KSG1 for best bias-variance trade-off.
- **Estimator choice:** KSG preferred over Gaussian to capture non-linear relationships; stock returns typically exhibit fatter tails than the Gaussian distribution.
- **Autocorrelation:** ACF analysis used to determine the Theiler window. Returns showed generally insignificant autocorrelation, so `DYN_CORR_EXCL = 0`.

---

## Key Findings

### Mutual Information (Cross-Sectional Analysis)

Strong dependencies exist among the traditional retail banks, while CBA and Macquarie exhibit distinctive dynamics.

1. **WBC, NAB, and ANZ pairs show the highest MI**, indicating strong stock price dependencies consistent with structural similarity among these three retail banks.
2. **CBA pairs show comparatively weaker dependence**, likely because CBA is viewed as a safer, more stable institution (largest ASX-listed stock, dominant market share, higher investor confidence).
3. **MQG pairs display the weakest and least consistent MI**, with some statistically insignificant periods, reflecting its more diversified business mix across asset management, infrastructure, and investment banking.

### Conditional Mutual Information (Cross-Sectional Analysis)

After controlling for market and macroeconomic factors, dependencies are substantially reduced.

- **CMI values drop substantially** once conditioning on broader market and macro factors, suggesting that much of the observed dependence is driven by common external forces.
- **Residual dependencies among the Big 4 retail banks remain**, suggesting sector-specific linkages in stock price returns beyond market-wide influences.
- **CMI values involving MQG are now commonly statistically insignificant**, consistent with its diversified business model — its price co-movement with other banks is largely market/macro related.

### Time-Series Analysis

Bank dependencies fluctuate with market conditions, intensifying during periods of stress.

- **MI peaks sharply during systemic stress events** (e.g. COVID-19, RBA rate hikes), reflecting transient surges in interbank price dependencies.
- **Conditioning on the ASX200 Ex-Top 20 significantly dampens these peaks**, showing that much of the co-movement is explained by broad market movements.
- **Conditioning on Australian Government Bond yields produces a smaller, time-varying reduction**, suggesting sensitivity to changing macroeconomic and policy environments.

---

## Conclusion

> **As shown by high MI and low CMI, share price dependence among Australia's major banks is primarily related to market and macroeconomic forces rather than sector-specific structural linkages.**

This places the result in the *moderate case* — systematic rather than idiosyncratic risk — with residual sector-specific dependencies remaining among the Big 4 retail banks.

**Limitations and Future Work:**
- **Temporal resolution:** Analysis is limited to daily data, which may overlook short-term intra-day dynamics.
- **Proxy selection:** Conditioning on only two broad proxies may simplify macroeconomic complexity. Expanding to credit spreads, funding costs, or housing data could improve the analysis.
- **Future directions:** Transfer entropy or effective connectivity graphs could map directional information flow among banks and macro variables.

---

## Repository Structure

```
BankingSectorAnalysis/
├── Bank_Analysis_vF.ipynb          # Full analysis notebook (final version)
├── BankAnalysis.py                 # Standalone script version
├── Zhao_Daniel_Presentation_vF.pdf # Presentation slides (downloadable)
├── mi_time_series.csv              # Pairwise MI time series
├── cmi_both_controls.csv           # CMI conditioned on both variables
├── cmi_interest_rates.csv          # CMI conditioned on bond index only
├── cmi_asx200.csv                  # CMI conditioned on ASX200 only
├── mi_pvalues.csv                  # MI significance p-values
├── cmi_pvalues.csv                 # CMI significance p-values
├── bank_mi_matrix.csv              # Static MI matrix
└── bank_cmi_matrix.csv             # Static CMI matrix
```

---

## Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn yfinance jpype1
```

You also need the **JIDT library** (`infodynamics.jar`), available at [https://github.com/jlizier/jidt](https://github.com/jlizier/jidt). Update the `jar_location` variable at the top of the notebook to point to your local copy.

### Running the Analysis

Open `Bank_Analysis_vF.ipynb` in Jupyter and run all cells. Pre-computed results are saved as CSV files and can be loaded directly to skip the computation step (which takes several minutes).

---

## References

- Australian Securities Exchange. (2025). *Top 50 by market capitalisation on the ASX*. ASX.
- Harre, M., & Bossomaier, T. (2009). Phase-transition-like behaviour of mutual information in financial markets. *Physical Review E, 79*(1), 016103.
- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Physical Review E, 69*(6), 066138.
- Kwon, O., & Yang, J.-S. (2008). Information flow between stock indices. *Europhysics Letters, 82*(6), 68003.
- Lizier, J. T. (2014). JIDT: An information-theoretic toolkit for studying the dynamics of complex systems. *Frontiers in Robotics and AI, 1*, 11.
- Marschinski, R., & Kantz, H. (2002). Analysing the information flow between financial time series. *The European Physical Journal B, 30*(2), 275–281.
- Reserve Bank of Australia. (n.d.). *Cash rate target – statistical tables*. RBA Statistics.
- S&P Dow Jones Indices. (2025). *S&P/ASX 200: Index overview*. S&P Global.
- Senate Economics References Committee. (2011). *Competition within the Australian banking sector* (Chapter 9: Four Pillars Policy). Parliament of Australia.
