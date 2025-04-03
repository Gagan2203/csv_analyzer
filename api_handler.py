import os
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
import json

from data_loader import DataLoader
from visualizer import Visualizer
from statistical_analyzer import StatisticalAnalyzer
from ml_analyzer import MLAnalyzer
from gemini_analyzer import GeminiAnalyzer

class APIHandler:
    """
    A class for handling API requests for the data analysis application.
    """
    
    def __init__(self, data_dir='data', static_dir='static'):
        """
        Initialize the APIHandler class.
        
        Args:
            data_dir (str): Directory to store uploaded data files
            static_dir (str): Directory to store static files
        """
        self.data_dir = data_dir
        self.static_dir = static_dir
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(static_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(static_dir, 'ml_outputs'), exist_ok=True)
        os.makedirs(os.path.join(static_dir, 'gemini_outputs'), exist_ok=True)
        
        # Initialize modules
        self.data_loader = DataLoader(data_dir=data_dir)
        self.visualizer = Visualizer(self.data_loader, output_dir=os.path.join(static_dir, 'images'))
        self.statistical_analyzer = StatisticalAnalyzer(self.data_loader)
        self.ml_analyzer = MLAnalyzer(self.data_loader, output_dir=os.path.join(static_dir, 'ml_outputs'))
        self.gemini_analyzer = GeminiAnalyzer(self.data_loader, output_dir=os.path.join(static_dir, 'gemini_outputs'))
    
    def handle_upload(self, file):
        """
        Handle file upload request.
        
        Args:
            file: The uploaded file object
            
        Returns:
            dict: Response with status and message
        """
        if file and file.filename.endswith('.csv'):
            try:
                # Save the file
                file_path = os.path.join(self.data_dir, file.filename)
                file.save(file_path)
                
                # Load the data
                success = self.data_loader.load_data_from_file(file_path)
                
                if success:
                    return {
                        'status': 'success',
                        'message': f'Successfully loaded {file.filename}',
                        'data_info': self.data_loader.get_data_info()
                    }
                else:
                    return {
                        'status': 'error',
                        'message': 'Failed to load data from file'
                    }
            except Exception as e:
                return {
                    'status': 'error',
                    'message': f'Error processing file: {str(e)}'
                }
        else:
            return {
                'status': 'error',
                'message': 'Please upload a CSV file'
            }
    
    def handle_data_info(self):
        """
        Handle data info request.
        
        Returns:
            dict: Response with data information
        """
        data_info = self.data_loader.get_data_info()
        
        if data_info:
            return {
                'status': 'success',
                'data_info': data_info,
                'column_info': self.data_loader.get_column_info(),
                'numeric_columns': self.data_loader.get_numeric_columns(),
                'categorical_columns': self.data_loader.get_categorical_columns(),
                'preview': self.data_loader.get_data_preview(rows=10)
            }
        else:
            return {
                'status': 'error',
                'message': 'No data loaded'
            }
    
    def handle_visualization(self, viz_type, params):
        """
        Handle visualization request.
        
        Args:
            viz_type (str): Type of visualization
            params (dict): Parameters for the visualization
            
        Returns:
            dict: Response with visualization results
        """
        if not self.data_loader.get_data() is not None:
            return {
                'status': 'error',
                'message': 'No data loaded'
            }
        
        try:
            result = None
            
            if viz_type == 'distribution':
                column = params.get('column')
                if column:
                    result = self.visualizer.create_distribution_plots(column)
            
            elif viz_type == 'correlation':
                columns = params.get('columns')
                result = self.visualizer.create_correlation_heatmap(columns)
            
            elif viz_type == 'categorical':
                column = params.get('column')
                if column:
                    result = self.visualizer.create_categorical_plot(column)
            
            elif viz_type == 'scatter':
                x_column = params.get('x_column')
                y_column = params.get('y_column')
                if x_column and y_column:
                    result = self.visualizer.create_scatter_plot(x_column, y_column)
            
            elif viz_type == 'group_by':
                group_column = params.get('group_column')
                agg_column = params.get('agg_column')
                agg_func = params.get('agg_func', 'mean')
                if group_column and agg_column:
                    result = self.visualizer.create_group_by_plot(group_column, agg_column, agg_func)
            
            elif viz_type == 'pairplot':
                columns = params.get('columns')
                hue = params.get('hue')
                n_samples = params.get('n_samples', 1000)
                result = self.visualizer.create_pairplot(columns, hue, n_samples)
            
            elif viz_type == 'timeseries':
                date_column = params.get('date_column')
                value_column = params.get('value_column')
                freq = params.get('freq')
                if date_column and value_column:
                    result = self.visualizer.create_time_series_plot(date_column, value_column, freq)
            
            if result:
                return {
                    'status': 'success',
                    'result': result
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to create visualization'
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error creating visualization: {str(e)}'
            }
    
    def handle_statistics(self, stat_type, params):
        """
        Handle statistics request.
        
        Args:
            stat_type (str): Type of statistical analysis
            params (dict): Parameters for the analysis
            
        Returns:
            dict: Response with statistical results
        """
        if not self.data_loader.get_data() is not None:
            return {
                'status': 'error',
                'message': 'No data loaded'
            }
        
        try:
            result = None
            
            if stat_type == 'descriptive':
                columns = params.get('columns')
                result = self.statistical_analyzer.get_descriptive_statistics(columns)
            
            elif stat_type == 'group_by':
                group_column = params.get('group_column')
                agg_column = params.get('agg_column')
                agg_func = params.get('agg_func', 'mean')
                if group_column and agg_column:
                    result = self.statistical_analyzer.get_group_by_statistics(group_column, agg_column, agg_func)
            
            elif stat_type == 'correlation':
                columns = params.get('columns')
                result = self.statistical_analyzer.get_correlation_matrix(columns)
            
            elif stat_type == 'summary_by_category':
                category_column = params.get('category_column')
                value_column = params.get('value_column')
                if category_column and value_column:
                    result = self.statistical_analyzer.get_summary_by_category(category_column, value_column)
            
            elif stat_type == 'quantiles':
                column = params.get('column')
                q = params.get('q')
                if column:
                    result = self.statistical_analyzer.get_quantiles(column, q)
            
            elif stat_type == 'value_counts':
                column = params.get('column')
                normalize = params.get('normalize', False)
                limit = params.get('limit', 20)
                if column:
                    result = self.statistical_analyzer.get_value_counts(column, normalize, limit)
            
            elif stat_type == 'missing_values':
                result = self.statistical_analyzer.get_missing_values_summary()
            
            elif stat_type == 'outliers':
                columns = params.get('columns')
                method = params.get('method', 'iqr')
                threshold = params.get('threshold', 1.5)
                result = self.statistical_analyzer.get_outliers_summary(columns, method, threshold)
            
            elif stat_type == 'distribution':
                column = params.get('column')
                bins = params.get('bins', 10)
                if column:
                    result = self.statistical_analyzer.get_data_distribution(column, bins)
            
            if result is not None:
                return {
                    'status': 'success',
                    'result': result
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to perform statistical analysis'
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error performing statistical analysis: {str(e)}'
            }
    
    def handle_ml(self, ml_type, params):
        """
        Handle machine learning request.
        
        Args:
            ml_type (str): Type of machine learning analysis
            params (dict): Parameters for the analysis
            
        Returns:
            dict: Response with machine learning results
        """
        if not self.data_loader.get_data() is not None:
            return {
                'status': 'error',
                'message': 'No data loaded'
            }
        
        try:
            result = None
            
            if ml_type == 'pca':
                columns = params.get('columns')
                n_components = params.get('n_components')
                result = self.ml_analyzer.perform_pca(columns, n_components)
            
            elif ml_type == 'clustering':
                columns = params.get('columns')
                n_clusters = params.get('n_clusters', 3)
                result = self.ml_analyzer.perform_clustering(columns, n_clusters)
            
            elif ml_type == 'regression':
                target_column = params.get('target_column')
                feature_columns = params.get('feature_columns')
                test_size = params.get('test_size', 0.2)
                if target_column:
                    result = self.ml_analyzer.train_regression_model(target_column, feature_columns, test_size)
            
            elif ml_type == 'classification':
                target_column = params.get('target_column')
                feature_columns = params.get('feature_columns')
                test_size = params.get('test_size', 0.2)
                if target_column:
                    result = self.ml_analyzer.train_classification_model(target_column, feature_columns, test_size)
            
            if result:
                return {
                    'status': 'success',
                    'result': result
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to perform machine learning analysis'
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error performing machine learning analysis: {str(e)}'
            }
    
    def handle_gemini(self, params):
        """
        Handle Gemini AI request.
        
        Args:
            params (dict): Parameters for the analysis
            
        Returns:
            dict: Response with Gemini AI results
        """
        if not self.data_loader.get_data() is not None:
            return {
                'status': 'error',
                'message': 'No data loaded'
            }
        
        try:
            api_key = params.get('api_key')
            analysis_type = params.get('analysis_type')
            custom_question = params.get('custom_question')
            
            if not api_key:
                return {
                    'status': 'error',
                    'message': 'API key is required'
                }
            
            if not analysis_type:
                return {
                    'status': 'error',
                    'message': 'Analysis type is required'
                }
            
            # Configure Gemini
            success = self.gemini_analyzer.configure(api_key)
            
            if not success:
                return {
                    'status': 'error',
                    'message': 'Failed to configure Gemini API'
                }
            
            # Perform analysis
            result = self.gemini_analyzer.analyze_data(analysis_type, custom_question)
            
            if result:
                return {
                    'status': 'success',
                    'result': result
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to perform Gemini analysis'
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error performing Gemini analysis: {str(e)}'
            }
    
    def handle_download(self, format_type):
        """
        Handle download request.
        
        Args:
            format_type (str): Format of the download (csv or json)
            
        Returns:
            Response: File download response
        """
        if not self.data_loader.get_data() is not None:
            return {
                'status': 'error',
                'message': 'No data loaded'
            }
        
        try:
            if format_type == 'csv':
                csv_data = self.data_loader.get_data_as_csv()
                return Response(
                    csv_data,
                    mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment;filename={self.data_loader.get_filename()}'}
                )
            
            elif format_type == 'json':
                json_data = self.data_loader.get_data_as_json()
                return Response(
                    json_data,
                    mimetype='application/json',
                    headers={'Content-Disposition': f'attachment;filename={self.data_loader.get_filename().replace(".csv", ".json")}'}
                )
            
            else:
                return {
                    'status': 'error',
                    'message': 'Invalid format type'
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error downloading data: {str(e)}'
            }
