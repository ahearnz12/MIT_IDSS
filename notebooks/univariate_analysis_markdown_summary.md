# Univariate Analysis Summary

## Customer Information Features

- **Year_Birth/Age**: Distribution shows most customers between 40-60 years old, with mean around 50-55. Few young customers (<30) or elderly (>75).

- **Income**: Right-skewed distribution with mean approximately $50-60K. Significant outliers in upper brackets (>$100K), suggesting high-value customer segment.

- **Education**: Higher education dominates - "Graduation" most common, followed by "PhD" and "Master". "Basic" education forms smallest segment.

- **Marital_Status**: "Married" is dominant (60-65%), followed by smaller segments of "Single", "Together", "Divorced", and "Widow".

- **Kidhome/Teenhome**: Most households (70-80%) have 0-1 children. Very few households have more than 1 child.

- **Recency**: Distribution relatively uniform with most customers purchasing within past 30-60 days. Long tail of dormant customers (60+ days).

- **Complain**: Over 95% of customers have no complaints, suggesting generally positive customer experience.

## Spending Information Features

- All spending features show right-skewed distributions, indicating a small segment of high-value customers.
- **Wine spending** has highest average and maximum values.
- **Meat products** are second highest spending category.
- **Sweet products** and **fruits** show lower average spending.
- **Gold products** show extreme right skew with many zero values.

## Campaign Interaction Features

- All campaigns show low acceptance rates (5-15%).
- Very few customers responded to multiple campaigns.
- **NumDealsPurchases**: Most customers make 1-3 deal purchases, with distribution peaking at 2-3 deals.

## Shopping Behavior Features

- **Store purchases** are most common (highest average), followed by web purchases.
- **Catalog purchases** have lowest average.
- **Web visits** show most customers visit 1-6 times monthly, with few visiting more than 10 times.

## Key Correlations

- **Strong positive correlations** between different spending categories (0.6-0.8).
- **Negative correlation** between children in household and luxury spending.
- **Positive correlation** between income and all spending categories.
- **Age** shows weak positive correlation with wine spending.
- **Web visits** negatively correlates with overall spending (browsers vs. buyers).

## Implications for Clustering

The univariate analysis reveals several patterns suggesting natural segments in the customer base:
1. Income-based segments (low, medium, high)
2. Family life stage segments (with/without children)
3. Spending behavior segments (luxury vs. essential purchasers)
4. Channel preference segments (store vs. online shoppers)

These patterns provide a strong foundation for customer segmentation and will guide feature selection for clustering algorithms.
