# Function to perform univariate analysis on all features
def perform_univariate_analysis(data):
    """
    Performs univariate analysis on all features in the dataset.
    
    Parameters:
    data (DataFrame): The dataset to analyze
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Identify feature types
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    
    # Remove ID from analysis if present
    if 'ID' in numeric_features:
        numeric_features.remove('ID')
    
    # Analyze numeric features
    print("ANALYZING NUMERIC FEATURES:")
    print("-" * 50)
    
    for feature in numeric_features:
        print(f"\nAnalysis for: {feature}")
        print("-" * 30)
        
        # Print basic statistics
        stats = data[feature].describe()
        print(f"Basic statistics:\n{stats}")
        
        # Create a figure with histogram and boxplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram with KDE
        sns.histplot(data[feature], kde=True, ax=ax1)
        ax1.set_title(f'Distribution of {feature}')
        
        # Boxplot
        sns.boxplot(y=data[feature], ax=ax2)
        ax2.set_title(f'Boxplot of {feature}')
        
        plt.tight_layout()
        plt.show()
        
        # Check for skewness
        skewness = data[feature].skew()
        if abs(skewness) < 0.5:
            print(f"The distribution is approximately symmetric (skewness: {skewness:.2f})")
        elif skewness > 0:
            print(f"The distribution is right-skewed (skewness: {skewness:.2f})")
        else:
            print(f"The distribution is left-skewed (skewness: {skewness:.2f})")
        
        # Calculate outliers using IQR method
        q1 = stats['25%']
        q3 = stats['75%']
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)][feature]
        
        print(f"Number of potential outliers: {len(outliers)}")
        print(f"Percentage of outliers: {(len(outliers)/len(data))*100:.2f}%")
        print("-" * 30)
    
    # Analyze categorical features
    print("\n\nANALYZING CATEGORICAL FEATURES:")
    print("-" * 50)
    
    for feature in categorical_features:
        print(f"\nAnalysis for: {feature}")
        print("-" * 30)
        
        # Value counts and percentages
        value_counts = data[feature].value_counts()
        value_percentage = data[feature].value_counts(normalize=True) * 100
        
        print(f"Value counts:\n{value_counts}")
        print(f"\nPercentages:\n{value_percentage}")
        
        # Plot bar chart
        plt.figure(figsize=(12, 6))
        ax = sns.countplot(x=feature, data=data)
        
        # Rotate x labels if needed
        plt.xticks(rotation=45)
        
        # Add percentage labels
        total = len(data)
        for p in ax.patches:
            percentage = 100 * p.get_height() / total
            ax.annotate(f'{percentage:.1f}%', 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'bottom',
                       xytext = (0, 5), textcoords = 'offset points')
        
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
        plt.show()
        print("-" * 30)
    
    # Create a correlation heatmap for numeric features
    print("\n\nCORRELATION ANALYSIS:")
    print("-" * 50)
    
    # Filter numeric features for correlation
    corr_features = [f for f in numeric_features if f not in ['ID']]
    
    # Create correlation matrix
    plt.figure(figsize=(16, 12))
    corr_matrix = data[corr_features].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, mask=mask)
    plt.title('Correlation Heatmap of Numeric Features', fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    print("\nUnivariate analysis complete!")

# Example usage:
# perform_univariate_analysis(data)
