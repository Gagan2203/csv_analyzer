import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np

class Visualizer:
    """
    A class for creating various data visualizations.
    """
    
    def __init__(self, data_loader):
        """
        Initialize the Visualizer class.
        
        Args:
            data_loader: An instance of the DataLoader class
        """
        self.data_loader = data_loader
    
    def create_distribution_plots(self, column):
        """
        Create distribution plots (histogram and box plot) for a numeric column.
        
        Args:
            column (str): The name of the numeric column to visualize
            
        Returns:
            tuple: A tuple containing two matplotlib figures (histogram and box plot)
        """
        df = self.data_loader.get_data()
        
        if df is not None and column in df.columns:
            # Create histogram
            hist_fig, hist_ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[column].dropna(), kde=True, ax=hist_ax)
            hist_ax.set_title(f'Distribution of {column}')
            
            # Create box plot
            box_fig, box_ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(y=df[column].dropna(), ax=box_ax)
            box_ax.set_title(f'Box Plot of {column}')
            
            return hist_fig, box_fig
        
        return None, None
    
    def create_correlation_heatmap(self, columns=None):
        """
        Create a correlation heatmap for numeric columns.
        
        Args:
            columns (list, optional): List of column names to include in the heatmap.
                If None, all numeric columns will be used.
                
        Returns:
            matplotlib.figure.Figure: The correlation heatmap figure
        """
        df = self.data_loader.get_data()
        
        if df is not None:
            if columns is None:
                columns = self.data_loader.get_numeric_columns()
            
            if len(columns) > 1:
                corr_matrix = df[columns].corr()
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                ax.set_title('Correlation Matrix')
                return fig
        
        return None
    
    def create_categorical_plot(self, column):
        """
        Create a bar plot for a categorical column.
        
        Args:
            column (str): The name of the categorical column to visualize
            
        Returns:
            matplotlib.figure.Figure: The bar plot figure
        """
        df = self.data_loader.get_data()
        
        if df is not None and column in df.columns:
            # Count values
            value_counts = df[column].value_counts().sort_values(ascending=False)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Limit to top 20 categories if there are too many
            if len(value_counts) > 20:
                value_counts = value_counts.head(20)
                ax.set_title(f'Top 20 Categories in {column}')
            else:
                ax.set_title(f'Categories in {column}')
            
            # Create bar plot
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            return fig
        
        return None
    
    def create_scatter_plot(self, x_column, y_column):
        """
        Create a scatter plot for two numeric columns.
        
        Args:
            x_column (str): The name of the column for the x-axis
            y_column (str): The name of the column for the y-axis
            
        Returns:
            matplotlib.figure.Figure: The scatter plot figure
        """
        df = self.data_loader.get_data()
        
        if df is not None and x_column in df.columns and y_column in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x=x_column, y=y_column, ax=ax)
            ax.set_title(f'Scatter Plot: {x_column} vs {y_column}')
            return fig
        
        return None
    
    def create_group_by_plot(self, group_column, agg_column, agg_func='mean'):
        """
        Create a bar plot for grouped data.
        
        Args:
            group_column (str): The column to group by
            agg_column (str): The column to aggregate
            agg_func (str): The aggregation function to use
            
        Returns:
            tuple: A tuple containing the grouped data and the bar plot figure
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
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Limit to top 20 categories if there are too many
            if len(grouped) > 20:
                grouped = grouped.head(20)
                ax.set_title(f'Top 20 {group_column} by {agg_func} of {agg_column}')
            else:
                ax.set_title(f'{group_column} by {agg_func} of {agg_column}')
            
            # Create bar plot
            sns.barplot(x=grouped.index, y=grouped.values, ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            return grouped, fig
        
        return None, None
    
    def create_pairplot(self, columns=None, hue=None, n_samples=1000):
        """
        Create a pairplot for multiple numeric columns.
        
        Args:
            columns (list, optional): List of column names to include in the pairplot.
                If None, all numeric columns will be used (up to 5).
            hue (str, optional): Column name to use for color encoding.
            n_samples (int, optional): Number of samples to use for the pairplot.
                
        Returns:
            seaborn.axisgrid.PairGrid: The pairplot figure
        """
        df = self.data_loader.get_data()
        
        if df is not None:
            if columns is None:
                columns = self.data_loader.get_numeric_columns()
                # Limit to 5 columns to avoid excessive computation
                columns = columns[:min(5, len(columns))]
            
            if len(columns) >= 2:
                # Sample data if it's large
                if len(df) > n_samples:
                    sample_df = df.sample(n_samples, random_state=42)
                else:
                    sample_df = df
                
                # Create pairplot
                pairplot = sns.pairplot(sample_df, vars=columns, hue=hue, height=2.5)
                pairplot.fig.suptitle('Pairwise Relationships', y=1.02)
                plt.tight_layout()
                
                return pairplot
        
        return None
    
    def create_time_series_plot(self, date_column, value_column, freq=None):
        """
        Create a time series plot.
        
        Args:
            date_column (str): The name of the date/time column
            value_column (str): The name of the value column to plot
            freq (str, optional): Frequency for resampling (e.g., 'D', 'M', 'Y')
            
        Returns:
            matplotlib.figure.Figure: The time series plot figure
        """
        df = self.data_loader.get_data()
        
        if df is not None and date_column in df.columns and value_column in df.columns:
            try:
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                    df[date_column] = pd.to_datetime(df[date_column])
                
                # Set date as index
                ts_df = df.set_index(date_column)
                
                # Resample if frequency is specified
                if freq is not None:
                    ts_df = ts_df[value_column].resample(freq).mean()
                else:
                    ts_df = ts_df[value_column]
                
                # Create figure
                fig, ax = plt.subplots(figsize=(12, 6))
                ts_df.plot(ax=ax)
                ax.set_title(f'Time Series: {value_column}')
                ax.set_ylabel(value_column)
                plt.tight_layout()
                
                return fig
            except Exception as e:
                st.error(f"Error creating time series plot: {e}")
        
        return None
