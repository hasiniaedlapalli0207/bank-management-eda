# Bank Marketing Campaign Analysis (EDA + Baseline ML)

## Project Objective
A bank ran phone campaigns to sell term deposits, and only about 11.7% of customers subscribed. This project explores customer and campaign data to find **which customers and which campaign practices are linked to subscription**, so future campaigns can be better targeted.

## Dataset
- **Source:** UCI Bank Marketing Dataset - https://archive.ics.uci.edu/dataset/222/bank+marketing
- **Size:** 45,211 customers, 16 features + 1 target (`y` = subscribed yes/no)
- **Context:** Portuguese bank, direct phone marketing campaigns (2008-2010)
- **Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Data Cleaning (key decisions)
| Issue | Decision | Reason |
|---|---|---|
| "unknown" in job, education, contact, poutcome | Kept as its own category | Dropping would lose many rows (poutcome is 81.7% unknown) |
| `pdays = -1` | Created `previously_contacted` flag | -1 is a code for "never contacted", not a real number |
| Target `y` (yes/no) | Converted to 1/0 | Mean of `y` = subscription rate |
| `balance` outliers (4,729 rows, 10.5%) and negatives (3,766) | Kept | Real customers; used medians and log scale |
| `age` | Binned into age groups | Easier segment comparison |
| `duration` | Used in EDA, excluded from ML | Known only after the call (data leakage) |

## Analysis Performed
- Univariate: target distribution, age, balance (log scale), campaign contacts, contact type
- Bivariate: subscription rate by age group, job, education, marital status, contact method, previous outcome, number of contacts
- Box plots: balance, duration and campaign by subscription
- Scatter plots: age vs balance, campaign vs duration
- Multivariate: job x education heatmap, correlation heatmap
- Seasonality: calls vs subscription rate by month
- ML: Logistic Regression baseline (train/test split, scaling, class balancing, precision/recall/F1, feature importance)

## Key Findings
1. **Only 11.7% subscribed**, so the data is heavily imbalanced.
2. **Age shows a U-shape:** under-25 (25.6%) and 65+ (42.1%) subscribe far more than ages 35-54 (about 9-10%). These two groups have only about 810 customers each.
3. **Students (28.7%) and retired customers (22.8%)** lead; blue-collar workers are lowest (about 7%).
4. **Education matters moderately:** tertiary 15.0% > secondary 10.6% > primary 8.6%.
5. **Previous campaign success is the strongest signal:** 64.7% subscribed again, about 5.5x the average.
6. **Diminishing returns on calls:** rate falls from 14.6% (1 contact) to 3.9% (11+ contacts).
7. **Contact method:** cellular 14.9%, telephone 13.4%, unknown only 4.1%.
8. **Balance:** subscribers have a higher median balance (733 vs 417 euros), but the groups overlap heavily.
9. **Seasonality:** May had about 30% of all calls but a low rate (about 7%), while March (52.0%), September, October and December had few calls but very high rates.

## Machine Learning Results (Logistic Regression)
`duration` was excluded to avoid data leakage. On the 9,043-customer test set:

| Metric (class "yes") | Value |
|---|---|
| Accuracy | 0.756 |
| Precision | 0.27 |
| Recall | 0.62 |
| F1-score | 0.37 |

The model found 659 of 1,058 actual subscribers. Customers it flagged subscribed at about 27%, roughly **2.3x the 11.7% baseline**. Accuracy is lower than an "always say no" model (about 88%), but that model finds zero subscribers, so accuracy alone is misleading here. The model's strongest features (previous success, unknown contact, number of contacts, retired status) agree with the EDA findings.

## Business Recommendations
- Re-contact **previous successes first** (64.7% conversion).
- Prioritise **students, retirees and tertiary-educated** customers.
- **Cap follow-ups at about 3 calls**; returns drop sharply afterwards.
- Prefer **cellular** contact and fix how contact method is recorded.
- Test shifting call volume from low-yield months (May) toward high-yield ones.

## Limitations
- Data from one bank, 2008-2010; patterns may not hold today.
- Findings show **association, not causation** (e.g. hard-to-convert customers may simply be called more often).
- Some segments and months have small samples (e.g. March, December, student x primary).
- No year column, so monthly results are a seasonal pattern, not a true time series.
- Logistic Regression is a linear baseline and cannot capture non-linear patterns such as the age U-shape.

## Repository Structure
```
bank-marketing-eda/
 |-- bank_marketing_eda.ipynb
 |-- README.md
 |-- EDA_Report.pdf
 |-- Presentation.pptx
 `-- (dataset link above)
```

## How to Run
1. Download the dataset from the UCI link above (use `bank-full.csv`, separator `;`) or load it with `ucimlrepo` (id=222).
2. Open the notebook in Jupyter or Google Colab and run all cells.
