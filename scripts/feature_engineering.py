# scripts/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import os

class FeatureEngineering:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, low_memory=False)

    def clean_data(self):
        print("🧹 Cleaning data...")
        self.data.replace(['', 'Unknown'], np.nan, inplace=True)

        numeric_cols = self.data.select_dtypes(include=['number']).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns

        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            self.data[col].fillna(self.data[col].median(), inplace=True)

        for col in categorical_cols:
            self.data[col].fillna(self.data[col].mode()[0], inplace=True)

    def process_dates(self):
        print("🕒 Processing date columns...")
        if 'VehicleIntroDate' in self.data.columns:
            self.data['VehicleIntroDate'] = pd.to_datetime(self.data['VehicleIntroDate'], errors='coerce')
            self.data['VehicleIntroYear'] = self.data['VehicleIntroDate'].dt.year
            self.data['VehicleAge'] = 2025 - self.data['VehicleIntroYear']

    def engineer_features(self):
        print("⚙️ Engineering new features...")
        if 'CapitalOutstanding' in self.data.columns and 'SumInsured' in self.data.columns:
            self.data['CapitalOutstanding'] = pd.to_numeric(self.data['CapitalOutstanding'], errors='coerce')
            self.data['SumInsured'] = pd.to_numeric(self.data['SumInsured'], errors='coerce')
            self.data['OutstandingRatio'] = self.data['CapitalOutstanding'] / (self.data['SumInsured'] + 1)

    def simplify_categories(self, threshold=0.01):
        print("📉 Simplifying rare categories...")
        cat_cols = self.data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            freq = self.data[col].value_counts(normalize=True)
            rare_labels = freq[freq < threshold].index
            self.data[col] = self.data[col].apply(lambda x: 'Other' if x in rare_labels else x)

    def encode_categorical(self):
        print("🔠 Encoding categorical variables...")
        cat_cols = self.data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if self.data[col].nunique() < 10:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(self.data[[col]])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
                self.data = pd.concat([self.data.drop(columns=[col]), encoded_df], axis=1)
            else:
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col])

    def scale_numerical(self):
        print("📏 Scaling numerical features...")
        scaler = StandardScaler()
        numeric_cols = self.data.select_dtypes(include=['number']).columns
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])

    def balance_data(self, target_column):
        print("⚖️ Balancing data with SMOTE...")
        if self.data[target_column].nunique() > 2:
            print("⚠️ Skipping SMOTE — target is continuous (not for regression).")
            return
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        self.data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name=target_column)], axis=1)

    def drop_irrelevant_columns(self, columns_to_drop):
        print("🧺 Dropping irrelevant columns...")
        self.data.drop(columns=[col for col in columns_to_drop if col in self.data.columns], axis=1, inplace=True)

    def process(self, target_column):
        self.clean_data()
        self.process_dates()
        self.engineer_features()
        self.simplify_categories()
        self.encode_categorical()
        self.scale_numerical()
        self.balance_data(target_column)
        self.drop_irrelevant_columns(['VehicleIntroDate'])
        return self.data


# ✅ Run this block when executing the script
if __name__ == "__main__":
    # Change the target_column here if needed
    input_path = 'data/MachineLearningRating_v3.csv'
    output_path = 'data/processed_data.csv'
    target_column = 'TotalClaims'  # or 'ClaimMade' for classification

    processor = FeatureEngineering(data_path=input_path)
    processed_data = processor.process(target_column=target_column)
    processed_data.to_csv(output_path, index=False)

    print(f"\n✅ Processed data saved to {output_path}")
