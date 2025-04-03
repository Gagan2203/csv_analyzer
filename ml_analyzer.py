import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import base64

class MLAnalyzer:
    """
    A class for performing machine learning analysis on data.
    """
    
    def __init__(self, data_loader):
        """
        Initialize the MLAnalyzer class.
        
        Args:
            data_loader: An instance of the DataLoader class
        """
        self.data_loader = data_loader
    
    def perform_pca(self, columns=None, n_components=None):
        """
        Perform Principal Component Analysis (PCA) on numeric columns.
        
        Args:
            columns (list, optional): List of column names to include.
                If None, all numeric columns will be used.
            n_components (int, optional): Number of principal components to compute.
                If None, min(n_samples, n_features) will be used.
                
        Returns:
            dict: Dictionary containing PCA results
        """
        df = self.data_loader.get_data()
        
        if df is not None:
            if columns is None:
                columns = self.data_loader.get_numeric_columns()
            
            if len(columns) >= 2:
                # Prepare data for PCA
                pca_data = df[columns].dropna()
                
                if len(pca_data) > 0:
                    # Standardize the data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(pca_data)
                    
                    # Determine number of components
                    if n_components is None:
                        n_components = min(len(pca_data), len(columns))
                    else:
                        n_components = min(n_components, min(len(pca_data), len(columns)))
                    
                    # Apply PCA
                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(scaled_data)
                    
                    # Create result dictionary
                    result = {
                        'pca': pca,
                        'pca_result': pca_result,
                        'explained_variance': pca.explained_variance_ratio_ * 100,
                        'loadings': pd.DataFrame(
                            pca.components_.T,
                            columns=[f'PC{i+1}' for i in range(n_components)],
                            index=columns
                        ),
                        'scaler': scaler,
                        'columns': columns
                    }
                    
                    return result
        
        return None
    
    def create_pca_plots(self, pca_result):
        """
        Create plots for PCA results.
        
        Args:
            pca_result (dict): Dictionary containing PCA results from perform_pca()
            
        Returns:
            dict: Dictionary containing PCA plots
        """
        if pca_result is not None:
            plots = {}
            
            # Explained variance plot
            fig_var, ax_var = plt.subplots(figsize=(10, 6))
            explained_var = pca_result['explained_variance']
            ax_var.bar(range(1, len(explained_var) + 1), explained_var)
            ax_var.set_xlabel('Principal Component')
            ax_var.set_ylabel('Explained Variance (%)')
            ax_var.set_title('Explained Variance by Principal Components')
            ax_var.set_xticks(range(1, len(explained_var) + 1))
            plots['explained_variance'] = fig_var
            
            # Scatter plot of first two components
            if pca_result['pca_result'].shape[1] >= 2:
                fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
                scatter = ax_scatter.scatter(
                    pca_result['pca_result'][:, 0],
                    pca_result['pca_result'][:, 1],
                    alpha=0.5
                )
                ax_scatter.set_title('PCA: First Two Principal Components')
                ax_scatter.set_xlabel(f'PC1 ({pca_result["explained_variance"][0]:.2f}%)')
                ax_scatter.set_ylabel(f'PC2 ({pca_result["explained_variance"][1]:.2f}%)')
                plots['scatter'] = fig_scatter
            
            # Feature importance for PC1
            fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
            loadings = pca_result['loadings'].copy()
            loadings.sort_values(by='PC1', key=abs, ascending=False)['PC1'].plot(kind='bar', ax=ax_imp)
            ax_imp.set_title('Feature Importance for PC1')
            ax_imp.set_ylabel('Loading Score')
            plt.tight_layout()
            plots['importance'] = fig_imp
            
            return plots
        
        return None
    
    def perform_clustering(self, columns=None, n_clusters=3):
        """
        Perform K-means clustering on numeric columns.
        
        Args:
            columns (list, optional): List of column names to include.
                If None, all numeric columns will be used.
            n_clusters (int, optional): Number of clusters to form.
                
        Returns:
            dict: Dictionary containing clustering results
        """
        df = self.data_loader.get_data()
        
        if df is not None:
            if columns is None:
                columns = self.data_loader.get_numeric_columns()
            
            if len(columns) >= 2:
                # Prepare data for clustering
                cluster_data = df[columns].dropna()
                
                if len(cluster_data) > 0:
                    # Standardize the data
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(cluster_data)
                    
                    # Apply K-means
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(scaled_data)
                    
                    # Add cluster labels to the data
                    cluster_df = cluster_data.copy()
                    cluster_df['Cluster'] = clusters
                    
                    # Calculate cluster centers
                    centers = pd.DataFrame(
                        scaler.inverse_transform(kmeans.cluster_centers_),
                        columns=columns
                    )
                    centers.index.name = 'Cluster'
                    
                    # Create result dictionary
                    result = {
                        'kmeans': kmeans,
                        'clusters': clusters,
                        'cluster_df': cluster_df,
                        'centers': centers,
                        'scaler': scaler,
                        'columns': columns
                    }
                    
                    return result
        
        return None
    
    def create_clustering_plots(self, clustering_result):
        """
        Create plots for clustering results.
        
        Args:
            clustering_result (dict): Dictionary containing clustering results from perform_clustering()
            
        Returns:
            dict: Dictionary containing clustering plots
        """
        if clustering_result is not None:
            plots = {}
            
            # Get data from result
            cluster_df = clustering_result['cluster_df']
            centers = clustering_result['centers']
            columns = clustering_result['columns']
            n_clusters = len(centers)
            
            # Scatter plot of first two features
            if len(columns) >= 2:
                fig_scatter, ax_scatter = plt.subplots(figsize=(10, 8))
                scatter = ax_scatter.scatter(
                    cluster_df[columns[0]],
                    cluster_df[columns[1]],
                    c=cluster_df['Cluster'],
                    cmap='viridis',
                    alpha=0.6
                )
                
                # Plot cluster centers
                ax_scatter.scatter(
                    centers[columns[0]],
                    centers[columns[1]],
                    c=range(n_clusters),
                    cmap='viridis',
                    marker='X',
                    s=200,
                    edgecolor='k'
                )
                
                ax_scatter.set_title(f'K-Means Clustering (k={n_clusters})')
                ax_scatter.set_xlabel(columns[0])
                ax_scatter.set_ylabel(columns[1])
                plt.colorbar(scatter, label='Cluster')
                plots['scatter'] = fig_scatter
            
            # Cluster distribution
            fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
            cluster_counts = cluster_df['Cluster'].value_counts().sort_index()
            cluster_counts.plot(kind='bar', ax=ax_dist)
            ax_dist.set_title('Number of Data Points in Each Cluster')
            ax_dist.set_xlabel('Cluster')
            ax_dist.set_ylabel('Count')
            plots['distribution'] = fig_dist
            
            # Parallel coordinates plot for cluster profiles
            if len(columns) >= 3:
                # Standardize data for parallel coordinates
                from sklearn.preprocessing import MinMaxScaler
                
                # Select a subset of columns if there are too many
                plot_columns = columns[:min(6, len(columns))]
                
                # Create a copy of the data with selected columns
                plot_df = cluster_df[plot_columns + ['Cluster']].copy()
                
                # Scale the data for better visualization
                scaler = MinMaxScaler()
                plot_df[plot_columns] = scaler.fit_transform(plot_df[plot_columns])
                
                # Create parallel coordinates plot
                fig_parallel, ax_parallel = plt.subplots(figsize=(12, 8))
                
                # Plot each cluster
                for i in range(n_clusters):
                    cluster_data = plot_df[plot_df['Cluster'] == i]
                    pd.plotting.parallel_coordinates(
                        cluster_data, 'Cluster', 
                        color=plt.cm.viridis(i / n_clusters),
                        alpha=0.3, ax=ax_parallel
                    )
                
                ax_parallel.set_title('Parallel Coordinates Plot of Clusters')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plots['parallel'] = fig_parallel
            
            return plots
        
        return None
    
    def create_download_link(self, df, filename="processed_data.csv"):
        """
        Create a download link for a DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame to download
            filename (str, optional): Name of the file to download
            
        Returns:
            str: HTML link for downloading the data
        """
        if df is not None:
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
        return None
    
    def train_regression_model(self, target_column, feature_columns=None, test_size=0.2):
        """
        Train a regression model.
        
        Args:
            target_column (str): The target column to predict
            feature_columns (list, optional): List of feature columns to use.
                If None, all numeric columns except the target will be used.
            test_size (float, optional): Proportion of the data to use for testing
            
        Returns:
            dict: Dictionary containing regression model results
        """
        df = self.data_loader.get_data()
        
        if df is not None and target_column in df.columns:
            # Get numeric columns if feature_columns is None
            if feature_columns is None:
                feature_columns = [col for col in self.data_loader.get_numeric_columns() 
                                  if col != target_column]
            
            if len(feature_columns) > 0:
                # Prepare data
                model_data = df[feature_columns + [target_column]].dropna()
                
                if len(model_data) > 0:
                    X = model_data[feature_columns]
                    y = model_data[target_column]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Train model
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Feature importance
                    feature_importance = pd.DataFrame({
                        'Feature': feature_columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # Create result dictionary
                    result = {
                        'model': model,
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2,
                        'feature_importance': feature_importance,
                        'target_column': target_column,
                        'feature_columns': feature_columns
                    }
                    
                    return result
        
        return None
    
    def create_regression_plots(self, regression_result):
        """
        Create plots for regression model results.
        
        Args:
            regression_result (dict): Dictionary containing regression results from train_regression_model()
            
        Returns:
            dict: Dictionary containing regression plots
        """
        if regression_result is not None:
            plots = {}
            
            # Actual vs Predicted plot
            fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
            ax_pred.scatter(regression_result['y_test'], regression_result['y_pred'], alpha=0.5)
            ax_pred.plot([regression_result['y_test'].min(), regression_result['y_test'].max()], 
                        [regression_result['y_test'].min(), regression_result['y_test'].max()], 
                        'k--', lw=2)
            ax_pred.set_xlabel('Actual')
            ax_pred.set_ylabel('Predicted')
            ax_pred.set_title(f'Actual vs Predicted {regression_result["target_column"]}')
            plots['actual_vs_predicted'] = fig_pred
            
            # Residuals plot
            residuals = regression_result['y_test'] - regression_result['y_pred']
            fig_resid, ax_resid = plt.subplots(figsize=(10, 6))
            ax_resid.scatter(regression_result['y_pred'], residuals, alpha=0.5)
            ax_resid.axhline(y=0, color='k', linestyle='--', lw=2)
            ax_resid.set_xlabel('Predicted')
            ax_resid.set_ylabel('Residuals')
            ax_resid.set_title('Residuals Plot')
            plots['residuals'] = fig_resid
            
            # Feature importance plot
            fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
            feature_importance = regression_result['feature_importance']
            sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax_imp)
            ax_imp.set_title('Feature Importance')
            plots['importance'] = fig_imp
            
            return plots
        
        return None
    
    def train_classification_model(self, target_column, feature_columns=None, test_size=0.2):
        """
        Train a classification model.
        
        Args:
            target_column (str): The target column to predict
            feature_columns (list, optional): List of feature columns to use.
                If None, all numeric columns except the target will be used.
            test_size (float, optional): Proportion of the data to use for testing
            
        Returns:
            dict: Dictionary containing classification model results
        """
        df = self.data_loader.get_data()
        
        if df is not None and target_column in df.columns:
            # Get numeric columns if feature_columns is None
            if feature_columns is None:
                feature_columns = [col for col in self.data_loader.get_numeric_columns() 
                                  if col != target_column]
            
            if len(feature_columns) > 0:
                # Prepare data
                model_data = df[feature_columns + [target_column]].dropna()
                
                if len(model_data) > 0:
                    X = model_data[feature_columns]
                    y = model_data[target_column]
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
                    )
                    
                    # Train model
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    class_report = classification_report(y_test, y_pred, output_dict=True)
                    
                    # Feature importance
                    feature_importance = pd.DataFrame({
                        'Feature': feature_columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # Create result dictionary
                    result = {
                        'model': model,
                        'X_train': X_train,
                        'X_test': X_test,
                        'y_train': y_train,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'accuracy': accuracy,
                        'classification_report': class_report,
                        'feature_importance': feature_importance,
                        'target_column': target_column,
                        'feature_columns': feature_columns,
                        'classes': np.unique(y)
                    }
                    
                    return result
        
        return None
    
    def create_classification_plots(self, classification_result):
        """
        Create plots for classification model results.
        
        Args:
            classification_result (dict): Dictionary containing classification results from train_classification_model()
            
        Returns:
            dict: Dictionary containing classification plots
        """
        if classification_result is not None:
            plots = {}
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(classification_result['y_test'], classification_result['y_pred'])
            
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            ax_cm.set_title('Confusion Matrix')
            
            # Set tick labels if classes are available
            if 'classes' in classification_result:
                classes = classification_result['classes']
                if len(classes) <= 10:  # Only show labels if not too many classes
                    ax_cm.set_xticklabels(classes)
                    ax_cm.set_yticklabels(classes)
            
            plots['confusion_matrix'] = fig_cm
            
            # Feature importance plot
            fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
            feature_importance = classification_result['feature_importance']
            sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax_imp)
            ax_imp.set_title('Feature Importance')
            plots['importance'] = fig_imp
            
            # Classification report visualization
            class_report = classification_result['classification_report']
            
            # Extract metrics for each class
            classes = []
            precision = []
            recall = []
            f1 = []
            
            for class_name, metrics in class_report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    classes.append(class_name)
                    precision.append(metrics['precision'])
                    recall.append(metrics['recall'])
                    f1.append(metrics['f1-score'])
            
            # Create DataFrame for plotting
            report_df = pd.DataFrame({
                'Class': classes,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
            
            # Melt the DataFrame for easier plotting
            report_melted = pd.melt(report_df, id_vars=['Class'], 
                                   value_vars=['Precision', 'Recall', 'F1-Score'],
                                   var_name='Metric', value_name='Score')
            
            # Create the plot
            fig_report, ax_report = plt.subplots(figsize=(12, 6))
            sns.barplot(x='Class', y='Score', hue='Metric', data=report_melted, ax=ax_report)
            ax_report.set_title('Classification Metrics by Class')
            ax_report.set_ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plots['class_report'] = fig_report
            
            return plots
        
        return None
