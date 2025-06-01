import pandas as pd

# Read the dataset from 'data.csv' into a variable named df
df = pd.read_csv('data/data.csv')

# Display the first few rows to verify the data was loaded correctly
print("First 5 rows of the dataset:")
print(df.head())

# Display information about the DataFrame
print("\nDataFrame Info:")
print(df.info())

# Display basic statistics about the numeric columns
print("\nBasic Statistics:")
print(df.describe())

"""
Note: For the quiz, the key part is just:
import pandas as pd
df = pd.read_csv('data.csv')

The dataset fields as described in the quiz:
- Age: Age of the individual applying for credit
- Sex: Gender of the applicant (values: Male, Female)
- Job: Job type (values: Skilled, Unskilled & Non-resident, Highly Skilled)
- Housing: Type of accommodation (values: Own, Free, Rent)
- Saving accounts: Savings status (values: Little, Moderate, Rich, Quite Rich)
- Checking account: Checking account balance (values: Little, Moderate, Rich)
- Credit amount: Total amount of credit being applied for (numeric)
- Purpose: Intended use of the credit (values: Radio/TV, Education, Furniture/Equipment, etc.)
"""
