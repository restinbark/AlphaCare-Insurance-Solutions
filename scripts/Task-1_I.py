
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Eda:
    """
    A class for performing Exploratory Data Analysis (EDA) and data cleaning.
    """
    def __init__(self, file_path):

        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        """
        Load data from the specified file path with a tab delimiter.
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        print(f"Loading data from {self.file_path}")
        try:
            data = pd.read_csv(self.file_path) 
            data.columns = data.columns.str.strip()  
            print("Data loaded successfully.")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame() 

    def inspect_data(self):
        """
        Display basic information and statistics about the dataset.
        """
        print("Data Info:")
        print(self.data.info())
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        print("\nFirst Few Rows:")
        print(self.data.head())

    def eda_summary(self):
        """
        Generate summary statistics and visualizations for the dataset.
        """
        print("Summary Statistics:")
        print(self.data.describe(include='all'))  
        
        numeric_data = self.data.select_dtypes(include=['number'])
        if numeric_data.empty:
            print("No numeric data available for correlation analysis.")
        else:
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(numeric_data.corr(), annot=True, cmap='viridis', fmt=".2f")
            plt.title("Feature Correlation")
            plt.show()

    def visualize_distribution(self, columns):
        """
        Visualize the distribution of specified columns using histograms.
        Args:
            columns (list): List of column names to visualize.
        """
        for col in columns:
            if col in self.data.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.data[col], kde=True, bins=30, color='pink')
                plt.title(f"Distribution of {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.grid(axis='y', linestyle='--', alpha=1)
                plt.show()
                plt.savefig(f"images/{col}_distribution.png")
            else:
                print(f"Column {col} not found in the dataset.")

    def visualize_relationship(self, x_col, y_col):
        """
        Visualize the relationship between two columns using a scatter plot.
        Args:
            x_col (str): Column for the x-axis.
            y_col (str): Column for the y-axis.
        """
        if x_col in self.data.columns and y_col in self.data.columns:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=self.data[x_col], y=self.data[y_col], alpha=0.7, color='purple')
            plt.title(f"{x_col} vs. {y_col}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.show()
            plt.savefig(f"images/{x_col}_vs_{y_col}.png")
        else:
            print("One or more columns not found in the dataset.")
    def generate_report(self):
        """
        Generate an EDA report summarizing key findings.
        """
        print("\nKey Insights:")
        print(f"Total rows: {len(self.data)}")
        print(f"Columns: {list(self.data.columns)}")
        if 'Dur. (ms)' in self.data.columns:
            print(f"Average session duration: {self.data['Dur. (ms)'].mean():.2f} ms")
        if 'Total DL (Bytes)' in self.data.columns and 'Total UL (Bytes)' in self.data.columns:
            total_data_volume = self.data['Total DL (Bytes)'].sum() + self.data['Total UL (Bytes)'].sum()
            print(f"Total data volume: {total_data_volume:.2e} Bytes")
        if 'MSISDN/Number' in self.data.columns:
            print(f"Number of unique users: {self.data['MSISDN/Number'].nunique()}")

    

