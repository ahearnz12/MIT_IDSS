import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# Load the data
data = pd.read_csv("../data/Customer_Personality_Segmentation.csv", sep="\t")

# Basic data cleaning
# Handle missing values in Income if any
if data["Income"].dtype == object:
    data["Income"] = data["Income"].str.replace("[$,]", "").astype(float)
data["Income"] = data["Income"].replace(0, np.nan)
data["Income"].fillna(data["Income"].mean(), inplace=True)

# Convert date column to datetime
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True)
data['Customer_Age'] = (pd.to_datetime('now') - data['Dt_Customer']).dt.days // 365
data['Age'] = pd.to_datetime('now').year - data['Year_Birth']

# Identify numeric and categorical features
numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

# Remove ID from numeric features as it's not meaningful for analysis
if 'ID' in numeric_features:
    numeric_features.remove('ID')

# Function to perform univariate analysis for numeric features
def analyze_numeric_feature(feature):
    plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 2)
    
    # Histogram
    ax1 = plt.subplot(gs[0, 0])
    sns.histplot(data[feature], kde=True, ax=ax1)
    ax1.set_title(f'Histogram of {feature}')
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    
    # Boxplot
    ax2 = plt.subplot(gs[0, 1])
    sns.boxplot(y=data[feature], ax=ax2)
    ax2.set_title(f'Boxplot of {feature}')
    
    # Summary statistics
    ax3 = plt.subplot(gs[1, :])
    stats = data[feature].describe().to_frame().T
    ax3.axis('off')
    stats_table = pd.DataFrame({
        'Mean': stats['mean'],
        'Median': stats['50%'],
        'Std Dev': stats['std'],
        'Min': stats['min'],
        'Max': stats['max'],
        'Q1 (25%)': stats['25%'],
        'Q3 (75%)': stats['75%'],
        'IQR': stats['75%'] - stats['25%'],
        'Skewness': data[feature].skew(),
        'Kurtosis': data[feature].kurtosis()
    }).round(2)
    
    # Display the stats table
    table = ax3.table(
        cellText=stats_table.values,
        colLabels=stats_table.columns,
        rowLabels=['Statistics'],
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0']*len(stats_table.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.suptitle(f'Univariate Analysis of {feature}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    # Print additional insights
    print(f"Insights for {feature}:")
    print(f"  - Range: {stats['min'][0]:.2f} to {stats['max'][0]:.2f}")
    
    # Check for outliers using IQR method
    q1 = stats['25%'][0]
    q3 = stats['75%'][0]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)][feature]
    print(f"  - Potential outliers: {len(outliers)} values outside range {lower_bound:.2f} to {upper_bound:.2f}")
    
    # Check distribution shape
    if abs(data[feature].skew()) < 0.5:
        print(f"  - Distribution is approximately symmetric (skewness: {data[feature].skew():.2f})")
    elif data[feature].skew() > 0:
        print(f"  - Distribution is right-skewed (skewness: {data[feature].skew():.2f})")
    else:
        print(f"  - Distribution is left-skewed (skewness: {data[feature].skew():.2f})")
    print("\n")

# Function to perform univariate analysis for categorical features
def analyze_categorical_feature(feature):
    plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 2)
    
    # Count plot
    ax1 = plt.subplot(gs[0, :])
    value_counts = data[feature].value_counts().reset_index()
    value_counts.columns = [feature, 'Count']
    sns.barplot(x=feature, y='Count', data=value_counts, ax=ax1)
    ax1.set_title(f'Distribution of {feature}')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on top of bars
    for p in ax1.patches:
        ax1.annotate(f'{int(p.get_height())}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom',
                   xytext = (0, 5), textcoords = 'offset points')
    
    # Frequency table
    ax2 = plt.subplot(gs[1, :])
    ax2.axis('off')
    
    # Calculate frequencies and percentages
    freq_table = data[feature].value_counts().reset_index()
    freq_table.columns = [feature, 'Count']
    freq_table['Percentage'] = freq_table['Count'] / freq_table['Count'].sum() * 100
    freq_table = freq_table.sort_values('Count', ascending=False)
    
    # Display the frequency table
    table = ax2.table(
        cellText=freq_table.values.round(2),
        colLabels=freq_table.columns,
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0']*len(freq_table.columns)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.suptitle(f'Univariate Analysis of {feature}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    # Print additional insights
    print(f"Insights for {feature}:")
    print(f"  - Number of unique values: {data[feature].nunique()}")
    print(f"  - Most common value: {data[feature].value_counts().index[0]} ({data[feature].value_counts().iloc[0]} occurrences, {data[feature].value_counts().iloc[0]/len(data)*100:.2f}%)")
    print(f"  - Least common value: {data[feature].value_counts().index[-1]} ({data[feature].value_counts().iloc[-1]} occurrences, {data[feature].value_counts().iloc[-1]/len(data)*100:.2f}%)")
    print("\n")

# Special handling for binary features
def analyze_binary_feature(feature):
    plt.figure(figsize=(12, 6))
    
    # Count plot
    value_counts = data[feature].value_counts().reset_index()
    value_counts.columns = [feature, 'Count']
    sns.barplot(x=feature, y='Count', data=value_counts)
    plt.title(f'Distribution of {feature}')
    plt.ylabel('Count')
    
    # Add count and percentage labels on top of bars
    total = len(data)
    for p in plt.gca().patches:
        percentage = 100 * p.get_height() / total
        plt.annotate(f'{int(p.get_height())} ({percentage:.1f}%)', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom',
                   xytext = (0, 5), textcoords = 'offset points')
    
    plt.tight_layout()
    plt.show()
    
    # Print insights
    print(f"Insights for {feature}:")
    print(f"  - Percentage of 1s: {data[feature].mean()*100:.2f}%")
    print(f"  - Percentage of 0s: {(1-data[feature].mean())*100:.2f}%")
    print("\n")

# Perform analysis for all features
print("========== UNIVARIATE ANALYSIS ==========\n")

# Analyze numeric features
print("===== NUMERIC FEATURES =====\n")
for feature in numeric_features:
    # Skip Year_Birth since we created Age
    if feature in ['Year_Birth', 'Customer_Age']:
        continue
        
    # Check if binary
    if set(data[feature].unique()).issubset({0, 1, 0.0, 1.0}):
        analyze_binary_feature(feature)
    else:
        analyze_numeric_feature(feature)

# Analyze categorical features
print("===== CATEGORICAL FEATURES =====\n")
for feature in categorical_features:
    if feature == 'Dt_Customer':  # Skip date feature
        continue
    analyze_categorical_feature(feature)

# Create a correlation heatmap for numeric features
plt.figure(figsize=(18, 14))
numeric_data = data[numeric_features].copy()
# Drop binary campaign features for better visualization
campaign_features = [col for col in numeric_data.columns if col.startswith('Accepted')]
numeric_data = numeric_data.drop(columns=campaign_features + ['ID', 'Year_Birth', 'Customer_Age'])
correlation = numeric_data.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, mask=mask)
plt.title('Correlation Heatmap of Numeric Features', fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Univariate analysis complete!")
