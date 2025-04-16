# --------------------------------------- preprocessing.py ---------------------------------------
# Goal: Preprocess credit approval dataset 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- STEP 1: Load the datasets -----------------------------------------------------
# Load application and credit history datasets
application_df = pd.read_csv("data/application_record.csv")
credit_df = pd.read_csv("data/credit_record.csv")

# ---------------- STEP 2: Create labels (target column) -----------------------------------------
# Generate a new column called "Approved"
# Rule: if the credit history includes any late payments (status 2–5), label = 0 (not approved), otherwise = 1
def create_labels(credit_df):
    def assign_label(group):
        if any(status in group.values for status in ['2', '3', '4', '5']):
            return 0
        return 1
    labels = credit_df.groupby('ID')['STATUS'].apply(assign_label).reset_index()
    labels.columns = ['ID', 'Approved']
    return labels

labels_df = create_labels(credit_df)

# ---------------- STEP 3: Merge with application data ------------------------------------------
# Join credit labels with demographic info using ID
merged_df = pd.merge(application_df, labels_df, on='ID', how='inner')

# ---------------- STEP 4: Drop irrelevant column(s) --------------------------------------------
# FLAG_MOBIL is dropped because it's always 1 for everyone (not useful)
merged_df = merged_df.drop(columns=['FLAG_MOBIL'])

# ---------------- STEP 5: Fill missing values --------------------------------------------------
# Any missing values in OCCUPATION_TYPE are replaced with 'Unknown'
merged_df['OCCUPATION_TYPE'] = merged_df['OCCUPATION_TYPE'].fillna('Unknown')

# ---------------- STEP 6: Turn words into numbers ----------------------------------------------
# All string columns are turned into numbers (using label encoding)
def label_encode(series):
    unique_vals = sorted(series.unique()) # Get all unique values in the column (e.g. Let’s say the column is ['Y', 'N', 'Y', 'Y', 'N'])
    encoding = {val: idx for idx, val in enumerate(unique_vals)} # Make a dictionary to assign numbers (unique_vals → ['N', 'Y'])
    return series.map(encoding), encoding # Convert text to numbers using the dictionary (encoding → {'N': 0, 'Y': 1}, series.map(encoding) → [1, 0, 1, 1, 0])

encoders = {}
for col in merged_df.select_dtypes(include='object').columns:
    merged_df[col], encoders[col] = label_encode(merged_df[col])

# ---------------- STEP 7: Normalize numerical values -------------------------------------------
# Apply Z-score normalization manually
# Every numerical column is standardized: (Subtract the mean/Divide by standard deviation)
def z_score_normalize(column):
    mean = np.mean(column)
    std = np.std(column)
    return (column - mean) / std

X = merged_df.drop(columns=['ID', 'Approved'])
y = merged_df['Approved']
X_scaled = X.apply(z_score_normalize)

# ---------------- STEP 8: Manual Train/Test Split ----------------------------------------------
# # 80% for training, 20% for testing (shuffled)
def train_test_split_manual(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    test_count = int(len(X) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = train_test_split_manual(X_scaled, y)

# ---------------- STEP 9: Output Summary ------------------------------------------------------
print("✅ Training shape:", X_train.shape)
print("✅ Test shape:", X_test.shape)
print("✅ Training label counts:\n", y_train.value_counts())
print("✅ Test label counts:\n", y_test.value_counts())

# ---------------- STEP 10: Test the preprocessing pipeline ------------------------------------
print("\n🟢 Sample of preprocessed X_train:")
print(X_train.head())

print("\n🟢 Sample of corresponding y_train labels:")
print(y_train.head())

# To make sure the data doesn't contain missing values or infinite values after preprocessing
print("\nAny NaNs in X_train?", X_train.isnull().any().any())
print("Any NaNs in X_test?", X_test.isnull().any().any())
print("Any Infs in X_train?", np.isinf(X_train.values).any())
print("Any Infs in X_test?", np.isinf(X_test.values).any())

# ---------------- STEP 11: Save the processed data --------------------------------------------
# Save cleaned and split data as CSV files
X_train.to_csv("data/processed_dataset/X_train.csv", index=False) #	Features for training
X_test.to_csv("data/processed_dataset/X_test.csv", index=False) # Features for testing
y_train.to_csv("data/processed_dataset/y_train.csv", index=False) #	Labels for training
y_test.to_csv("data/processed_dataset/y_test.csv", index=False) # Labels for testing

# Optional: Save full processed dataset (with labels)
processed_df = pd.concat([X_scaled, y], axis=1)
processed_df.to_csv("data/processed_dataset/processed_full_dataset.csv", index=False) # Full cleaned dataset (features + label)

print("\nPreprocessed data saved in: data/processed_dataset/")

