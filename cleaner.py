import pandas as pd

# Read the CSV file
df = pd.read_csv('joueurs_football_dataset.csv')

# Clean the Comp column by removing the country codes at the start
df['Comp'] = df['Comp'].str.replace(r'^[a-z]{2,3}\s+', '', regex=True)

# Save the cleaned data back to CSV
df.to_csv('cleaned_football_dataset.csv', index=False)
