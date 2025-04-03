import pandas as pd
import numpy as np
import streamlit as st

class StatisticalAnalyzer:
    """
    A class for performing statistical analysis on data.
    """
    
    def __init__(self, data_loader):
        """
        Initialize the StatisticalAnalyzer class.
        
        Args:
            data_loader: An instance of the DataLoader class
        """
        self.data_loader = data_loader
    
    def get_descriptive_statistics(self, columns=None):
        """
        Calculate descriptive statistics for numeric columns.
        
        Args:
            columns (list, optional): List of column names to include.
                If None, all numeric columns will be used.
                
        Returns:
            pandas.DataFrame: DataFrame containing descriptive statistics
        """
        df = self.data_loader.get_data()
        
        if df is not None:
            if columns is None:
                columns = self.data_loader.get_numeric_columns()
            
            if len(columns) > 0:
                return df[columns].describe().T
        
        return None
    
    def get_group_by_statistics(self, group_column, agg_column, agg_func='mean'):
        """
        Calculate statistics for grouped data.
        
        Args:
            group_column (str): The column to group by
            agg_column (str): The column to aggregate
            agg_func (str): The aggregation function to use
            
        Returns:
            pandas.DataFrame: DataFrame containing grouped statistics
        """
        df = self.data_loader.get_data()
        
        if df is not None and group_column in df.columns and agg_column in df.columns:
            # Map function name to pandas function
            agg_map = {
                "Mean": "mean",
                "Median": "median",
                "Sum": "sum",
                "Min": "min",
                "Max": "max",
                "Count": "count",
                "Std Dev": "std"
            }
            
            # Use the mapped function or the original if not in the map
            func = agg_map.get(agg_func, agg_func.lower())
            
            # Perform groupby operation
            grouped = df.groupby(group_column)[agg_column].agg(func).sort_values(ascending=False)
            
            return grouped.reset_index()
        
        return None
    
    def get_correlation_matrix(self, columns=None):
        """
        Calculate the correlation matrix for numeric columns.
        
        Args:
            columns (list, optional): List of column names to include.
                If None, all numeric columns will be used.
                
        Returns:
            pandas.DataFrame: DataFrame containing the correlation matrix
        """
        df = self.data_loader.get_data()
        
        if df is not None:
            if columns is None:
                columns = self.data_loader.get_numeric_columns()
            
            if len(columns) > 1:
                return df[columns].corr()
        
        return None
    
    def get_summary_by_category(self, category_column, value_column):
        """
        Calculate summary statistics grouped by a categorical column.
        
        Args:
            category_column (str): The categorical column to group by
            value_column (str): The numeric column to summarize
            
        Returns:
            pandas.DataFrame: DataFrame containing summary statistics by category
        """
        df = self.data_loader.get_data()
        
        if df is not None and category_column in df.columns and value_column in df.columns:
            return df.groupby(category_column)[value_column].agg([
                'count', 'mean', 'median', 'min', 'max', 'std'
            ]).sort_values('count', ascending=False)
        
        return None
    
    def get_quantiles(self, column, q=None):
        """
        Calculate quantiles for a numeric column.
        
        Args:
            column (str): The numeric column to analyze
            q (list, optional): List of quantiles to calculate.
                If None, default quantiles [0, 0.25, 0.5, 0.75, 1] will be used.
                
        Returns:
            pandas.Series: Series containing the quantiles
        """
        df = self.data_loader.get_data()
        
        if df is not None and column in df.columns:
            if q is None:
                q = [0, 0.25, 0.5, 0.75, 1]
            
            return df[column].quantile(q)
        
        return None
    
    def get_value_counts(self, column, normalize=False, limit=20):
        """
        Calculate value counts for a column.
        
        Args:
            column (str): The column to analyze
            normalize (bool, optional): Whether to return proportions instead of counts
            limit (int, optional): Maximum number of values to return
            
        Returns:
            pandas.Series: Series containing the value counts
        """
        df = self.data_loader.get_data()
        
        if df is not None and column in df.columns:
            counts = df[column].value_counts(normalize=normalize)
            
            if len(counts) > limit:
                return counts.head(limit)
            
            return counts
        
        return None
    
    def get_missing_values_summary(self):
        """
        Calculate summary of missing values for each column.
        
        Returns:
            pandas.DataFrame: DataFrame containing missing values summary
        """
        df = self.data_loader.get_data()
        
        if df is not None:
            # Calculate missing values
            missing = df.isnull().sum()
            missing_percent = (missing / len(df)) * 100
            
            # Create summary DataFrame
            missing_summary = pd.DataFrame({
                'Column': missing.index,
                'Missing Values': missing.values,
                'Missing (%)': missing_percent.values
            })
            
            # Sort by missing values (descending)
            missing_summary = missing_summary.sort_values('Missing Values', ascending=False)
            
            return missing_summary
        
        return None
    
    def get_outliers_summary(self, columns=None, method='iqr', threshold=1.5):
        """
        Identify outliers in numeric columns.
        
        Args:
            columns (list, optional): List of column names to analyze.
                If None, all numeric columns will be used.
            method (str, optional): Method to use for outlier detection.
                'iqr': Interquartile Range method
                'zscore': Z-score method
            threshold (float, optional): Threshold for outlier detection.
                For IQR method: values outside Q1 - threshold*IQR and Q3 + threshold*IQR
                For Z-score method: values with absolute Z-score > threshold
                
        Returns:
            pandas.DataFrame: DataFrame containing outlier summary
        """
        df = self.data_loader.get_data()
        
        if df is not None:
            if columns is None:
                columns = self.data_loader.get_numeric_columns()
            
            outlier_summary = []
            
            for col in columns:
                if method == 'iqr':
                    # IQR method
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    
                elif method == 'zscore':
                    # Z-score method
                    mean = df[col].mean()
                    std = df[col].std()
                    
                    if std == 0:  # Avoid division by zero
                        outliers = pd.Series([], dtype=float)
                    else:
                        z_scores = (df[col] - mean) / std
                        outliers = df[abs(z_scores) > threshold][col]
                
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                outlier_count = len(outliers)
                outlier_percent = (outlier_count / len(df)) * 100
                
                if outlier_count > 0:
                    outlier_min = outliers.min()
                    outlier_max = outliers.max()
                else:
                    outlier_min = None
                    outlier_max = None
                
                outlier_summary.append({
                    'Column': col,
                    'Outlier Count': outlier_count,
                    'Outlier (%)': outlier_percent,
                    'Min Outlier': outlier_min,
                    'Max Outlier': outlier_max
                })
            
            return pd.DataFrame(outlier_summary)
        
        return None
