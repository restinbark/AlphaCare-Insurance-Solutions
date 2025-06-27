
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder







class DataPreparation:
    def __init__(self, data_path):
        self.data = data_path
    
    def handle_missing_values(self):
        self.data.dropna(axis=1, how='all', inplace=True)
        self.data.dropna(thresh=self.data.shape[1] // 2, inplace=True)
    def encode_categorical_data(self):
        categorical_columns = self.data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            self.data[col] = self.data[col].astype(str).str.strip()  # Strip whitespaces
            if self.data[col].replace('', pd.NA).dropna().nunique() == 0:
                print(f"Skipping column '{col}' as it has no unique values.")
                continue
            if self.data[col].nunique() < 10:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_data = encoder.fit_transform(self.data[[col]])
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([col]))
                self.data = pd.concat([self.data, encoded_df], axis=1)
                self.data.drop(col, axis=1, inplace=True)
            else:
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col])

    
    def split_data(self, target_column):
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        
        if X.empty:
            raise ValueError("Feature set is empty after preprocessing.")
        if y.isnull().all():
            raise ValueError("Target column contains only missing values.")
        
        return train_test_split(X, y, test_size=0.3, random_state=42)
class Model:
    def __init__(self, model):
        self.model = model
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, y_test, y_pred):
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2
    
    def feature_importance(self, X_train):
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            feature_names = X_train.columns
            feature_importance = pd.DataFrame(
                {"Feature": feature_names, "Importance": importance}
            ).sort_values(by="Importance", ascending=False)
            print("Feature Importances:")
            print(feature_importance)
        else:
            print("Feature importance is not available for this model.")


class StatisticalModelingPipeline:
    def __init__(self, data_path, target_column):
        self.data_preparation = DataPreparation(data_path)
        self.target_column = target_column
    
    def run(self):
        # Data Preparation
        self.data_preparation.handle_missing_values()
        self.data_preparation.encode_categorical_data()
        X_train, X_test, y_train, y_test = self.data_preparation.split_data(self.target_column)
        
        # Model Building
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        }
        
        for name, model in models.items():
            print(f"Training {name}...")
            model_instance = Model(model)
            model_instance.train(X_train, y_train)
            y_pred = model_instance.predict(X_test)
            
            # Evaluation
            mse, r2 = model_instance.evaluate(y_test, y_pred)
            print(f"{name} - MSE: {mse:.2f}, R2: {r2:.2f}")
            
            # Feature Importance
            if name != "Linear Regression": 
                print(f"Feature importance for {name}:")
                model_instance.feature_importance(X_train)


if __name__ == "__main__":
    pipeline = StatisticalModelingPipeline(data_path=pd.read_csv("C:/Users/HP/OneDrive/Desktop/Tenx/AlphaCare_Insurance_solutions/data/machineLearningRating_v3.csv"), target_column='TotalClaims')
    pipeline.run()
