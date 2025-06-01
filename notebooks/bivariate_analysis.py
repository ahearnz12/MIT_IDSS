# Bivariate Analysis for Customer Personality Segmentation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Load the data
data = pd.read_csv("../data/Customer_Personality_Segmentation.csv", sep="\t")

# Basic data cleaning
# Handle missing values in Income if any
if data["Income"].dtype == object:
    data["Income"] = data["Income"].str.replace("[$,]", "").astype(float)
data["Income"] = data["Income"].replace(0, np.nan)
data["Income"].fillna(data["Income"].mean(), inplace=True)

# Convert date column to datetime and create age
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True)
data['Age'] = pd.to_datetime('now').year - data['Year_Birth']

# Calculate total spending
data['Total_Spending'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + \
                         data['MntFishProducts'] + data['MntSweetProducts'] + data['MntGoldProds']

# Calculate total purchases
data['Total_Purchases'] = data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases']

# Calculate total accepted campaigns
data['Total_Campaigns'] = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + \
                          data['AcceptedCmp4'] + data['AcceptedCmp5'] + data['Response']

# Define feature categories for structured analysis
demographic_features = ['Age', 'Income', 'Education', 'Marital_Status', 'Kidhome', 'Teenhome']
spending_features = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
                     'MntGoldProds', 'Total_Spending']
purchase_features = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumDealsPurchases',
                    'NumWebVisitsMonth', 'Total_Purchases']
campaign_features = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 
                    'Response', 'Total_Campaigns']
engagement_features = ['Recency', 'Complain']

# 1. Correlation Heatmap for numeric features
print("="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Select numeric features for correlation analysis
numeric_features = ['Age', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 
                   'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                   'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
                   'NumDealsPurchases', 'Total_Spending', 'Total_Purchases', 'Total_Campaigns']

plt.figure(figsize=(20, 16))
corr = data[numeric_features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True, fmt='.2f', annot_kws={"size": 8})
plt.title('Correlation Matrix of Numeric Features', fontsize=20)
plt.xticks(fontsize=10, rotation=45)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()

print("\nKey Correlation Insights:")
# Find strongest positive correlations (excluding self-correlations)
pos_corr = corr.unstack()
pos_corr = pos_corr[pos_corr < 1.0]  # Remove self-correlations
pos_corr = pos_corr.sort_values(ascending=False)
print("\nTop Positive Correlations:")
print(pos_corr.head(10))

# Find strongest negative correlations
neg_corr = corr.unstack()
neg_corr = neg_corr.sort_values(ascending=True)
print("\nTop Negative Correlations:")
print(neg_corr.head(10))

# 2. Demographic vs. Spending Relationships
print("\n" + "="*80)
print("DEMOGRAPHICS vs. SPENDING ANALYSIS")
print("="*80)

# Age vs. Different Spending Categories
plt.figure(figsize=(20, 12))
gs = GridSpec(2, 3)

spending_cats = ['MntWines', 'MntMeatProducts', 'MntFishProducts', 
                 'MntFruits', 'MntSweetProducts', 'MntGoldProds']

for i, cat in enumerate(spending_cats):
    ax = plt.subplot(gs[i//3, i%3])
    sns.scatterplot(x='Age', y=cat, data=data, alpha=0.6, ax=ax)
    
    # Add regression line
    sns.regplot(x='Age', y=cat, data=data, scatter=False, ax=ax, line_kws={"color": "red"})
    
    # Calculate correlation
    corr_val = data['Age'].corr(data[cat])
    ax.set_title(f'Age vs. {cat} (Correlation: {corr_val:.2f})')

plt.tight_layout()
plt.show()

# Income vs. Total Spending
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Income', y='Total_Spending', data=data, alpha=0.6, hue='Education')
plt.title('Income vs. Total Spending by Education Level')
plt.xlabel('Income')
plt.ylabel('Total Spending ($)')
plt.legend(title='Education', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Kids/Teens in household vs. Spending Categories
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Kids impact on spending
sns.boxplot(x='Kidhome', y='Total_Spending', data=data, ax=axes[0])
axes[0].set_title('Number of Kids vs. Total Spending')
axes[0].set_xlabel('Number of Kids in Household')
axes[0].set_ylabel('Total Spending ($)')

# Teens impact on spending
sns.boxplot(x='Teenhome', y='Total_Spending', data=data, ax=axes[1])
axes[1].set_title('Number of Teens vs. Total Spending')
axes[1].set_xlabel('Number of Teens in Household')
axes[1].set_ylabel('Total Spending ($)')

plt.tight_layout()
plt.show()

# Education vs. Spending
plt.figure(figsize=(14, 8))
ax = sns.boxplot(x='Education', y='Total_Spending', data=data)
ax.set_title('Education Level vs. Total Spending')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Marital Status vs. Spending
plt.figure(figsize=(14, 8))
ax = sns.boxplot(x='Marital_Status', y='Total_Spending', data=data)
ax.set_title('Marital Status vs. Total Spending')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Purchase Behavior Analysis
print("\n" + "="*80)
print("PURCHASE BEHAVIOR ANALYSIS")
print("="*80)

# Stacked bar chart - Purchase channel distribution by education
purchase_channels = data[['Education', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']]
purchase_channels_agg = purchase_channels.groupby('Education').mean().reset_index()
purchase_channels_agg = purchase_channels_agg.set_index('Education')
purchase_channels_agg.columns = ['Web', 'Catalog', 'Store']
purchase_channels_agg = purchase_channels_agg.div(purchase_channels_agg.sum(axis=1), axis=0)

plt.figure(figsize=(12, 8))
purchase_channels_agg.plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Purchase Channel Distribution by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Proportion of Purchases')
plt.legend(title='Purchase Channel')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Income vs. Web Visits
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Income', y='NumWebVisitsMonth', data=data, alpha=0.6)
plt.title('Income vs. Number of Web Visits per Month')
plt.xlabel('Income')
plt.ylabel('Web Visits per Month')
plt.tight_layout()
plt.show()

# Recency vs. Web Visits
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Recency', y='NumWebVisitsMonth', data=data, alpha=0.6)
plt.title('Recency (Days since Last Purchase) vs. Web Visits per Month')
plt.xlabel('Recency (Days)')
plt.ylabel('Web Visits per Month')
plt.tight_layout()
plt.show()

# 4. Campaign Response Analysis
print("\n" + "="*80)
print("CAMPAIGN RESPONSE ANALYSIS")
print("="*80)

# Campaign acceptance by education
campaign_edu = data.groupby('Education')[campaign_features[:6]].mean()
campaign_edu = campaign_edu * 100  # Convert to percentage

plt.figure(figsize=(14, 8))
campaign_edu.plot(kind='bar')
plt.title('Campaign Acceptance Rate by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Acceptance Rate (%)')
plt.legend(title='Campaign')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Campaign acceptance by income groups
data['Income_Group'] = pd.qcut(data['Income'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
campaign_income = data.groupby('Income_Group')[campaign_features[:6]].mean()
campaign_income = campaign_income * 100  # Convert to percentage

plt.figure(figsize=(14, 8))
campaign_income.plot(kind='bar')
plt.title('Campaign Acceptance Rate by Income Group')
plt.xlabel('Income Group')
plt.ylabel('Acceptance Rate (%)')
plt.legend(title='Campaign')
plt.tight_layout()
plt.show()

# Age vs. Total Campaign Acceptances
plt.figure(figsize=(12, 6))
sns.boxplot(x='Total_Campaigns', y='Age', data=data)
plt.title('Age Distribution by Number of Accepted Campaigns')
plt.xlabel('Number of Accepted Campaigns')
plt.ylabel('Age')
plt.tight_layout()
plt.show()

# 5. Spending Profile by Demographics (Facet Grid)
print("\n" + "="*80)
print("SPENDING PROFILE BY DEMOGRAPHICS")
print("="*80)

# Create a facet grid showing spending patterns by age group and education
data['Age_Group'] = pd.cut(data['Age'], bins=[0, 35, 50, 65, 100], 
                          labels=['18-35', '36-50', '51-65', '65+'])

# Create a subset of data for visualization
subset_data = data.melt(id_vars=['Age_Group', 'Education'], 
                       value_vars=spending_features[:-1],
                       var_name='Spending_Category', value_name='Amount')

plt.figure(figsize=(16, 10))
g = sns.catplot(data=subset_data, x='Spending_Category', y='Amount', 
              hue='Age_Group', col='Education', kind='bar',
              height=4, aspect=1.2, palette='viridis')
g.set_xticklabels(rotation=45)
g.set_titles("{col_name}")
g.fig.suptitle('Spending Patterns by Age Group and Education', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

# 6. Pair Plot for Key Features
print("\n" + "="*80)
print("KEY FEATURES PAIR PLOT")
print("="*80)

# Select key features for pair plot
key_features = ['Age', 'Income', 'Total_Spending', 'Total_Purchases', 
               'NumWebVisitsMonth', 'Recency', 'Total_Campaigns']

plt.figure(figsize=(16, 12))
sns.pairplot(data[key_features], diag_kind='kde', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot of Key Features', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()

print("\nBivariate analysis complete!")
