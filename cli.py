import argparse
import os
import sys

def main():
    """
    Command-line interface for CSV Data Analyzer.
    Provides options to run the application in different modes.
    """
    parser = argparse.ArgumentParser(description='CSV Data Analyzer - Command Line Interface')
    
    # Main command options
    parser.add_argument('--mode', type=str, choices=['web', 'cli', 'batch'], default='web',
                        help='Application mode: web (Flask interface), cli (command line), or batch (process files)')
    
    # Web mode options
    parser.add_argument('--port', type=int, default=5000, 
                        help='Port to run the web server on (web mode only)')
    parser.add_argument('--host', type=str, default='0.0.0.0', 
                        help='Host to run the web server on (web mode only)')
    parser.add_argument('--debug', action='store_true', 
                        help='Run in debug mode (web mode only)')
    
    # CLI mode options
    parser.add_argument('--input', type=str, 
                        help='Input CSV file path (cli/batch mode)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save output files (cli/batch mode)')
    parser.add_argument('--analysis', type=str, 
                        choices=['overview', 'visualize', 'stats', 'ml', 'all'],
                        help='Type of analysis to perform (cli/batch mode)')
    
    # Batch mode options
    parser.add_argument('--input-dir', type=str,
                        help='Directory containing CSV files to process (batch mode)')
    parser.add_argument('--gemini-key', type=str,
                        help='Gemini API key for AI analysis (cli/batch mode)')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    os.makedirs('static/ml_outputs', exist_ok=True)
    os.makedirs('static/gemini_outputs', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Run in the selected mode
    if args.mode == 'web':
        run_web_mode(args)
    elif args.mode == 'cli':
        run_cli_mode(args)
    elif args.mode == 'batch':
        run_batch_mode(args)

def run_web_mode(args):
    """Run the application in web mode with Flask."""
    from app import create_app
    
    print(f"Starting web server on http://{args.host}:{args.port}")
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)

def run_cli_mode(args):
    """Run the application in command-line mode."""
    if not args.input:
        print("Error: --input CSV file is required for CLI mode")
        sys.exit(1)
    
    if not args.analysis:
        print("Error: --analysis type is required for CLI mode")
        sys.exit(1)
    
    print(f"Running CLI mode analysis on {args.input}")
    
    # Import modules
    from modules.data_loader.data_loader import DataLoader
    from modules.visualization.visualizer import Visualizer
    from modules.statistics.statistical_analyzer import StatisticalAnalyzer
    from modules.ml_insights.ml_analyzer import MLAnalyzer
    from modules.gemini_integration.gemini_analyzer import GeminiAnalyzer
    
    # Initialize modules
    data_loader = DataLoader(data_dir='data')
    visualizer = Visualizer(data_loader, output_dir=os.path.join(args.output_dir, 'images'))
    statistical_analyzer = StatisticalAnalyzer(data_loader)
    ml_analyzer = MLAnalyzer(data_loader, output_dir=os.path.join(args.output_dir, 'ml_outputs'))
    
    # Load data
    success = data_loader.load_data_from_file(args.input)
    
    if not success:
        print(f"Error: Failed to load data from {args.input}")
        sys.exit(1)
    
    # Print data info
    data_info = data_loader.get_data_info()
    print("\n=== Data Information ===")
    print(f"Filename: {data_info['filename']}")
    print(f"Rows: {data_info['rows']}")
    print(f"Columns: {data_info['columns']}")
    print(f"Memory Usage: {data_info['memory_usage']}")
    print(f"Missing Values: {data_info['missing_values']}")
    
    # Run requested analysis
    if args.analysis in ['overview', 'all']:
        print("\n=== Data Overview ===")
        column_info = data_loader.get_column_info()
        print("\nColumn Information:")
        for col in column_info:
            print(f"  {col['Column']} ({col['Data Type']}): {col['Unique Values']} unique values, {col['Missing Values']} missing ({col['Missing (%)']}%)")
    
    if args.analysis in ['visualize', 'all']:
        print("\n=== Generating Visualizations ===")
        numeric_columns = data_loader.get_numeric_columns()
        categorical_columns = data_loader.get_categorical_columns()
        
        if numeric_columns:
            print(f"Creating distribution plots for {numeric_columns[0]}...")
            result = visualizer.create_distribution_plots(numeric_columns[0])
            print(f"  Saved to: {result['histogram']} and {result['boxplot']}")
            
            print("Creating correlation heatmap...")
            result = visualizer.create_correlation_heatmap()
            print(f"  Saved to: {result}")
        
        if categorical_columns:
            print(f"Creating categorical plot for {categorical_columns[0]}...")
            result = visualizer.create_categorical_plot(categorical_columns[0])
            print(f"  Saved to: {result}")
        
        if len(numeric_columns) >= 2:
            print(f"Creating scatter plot for {numeric_columns[0]} vs {numeric_columns[1]}...")
            result = visualizer.create_scatter_plot(numeric_columns[0], numeric_columns[1])
            print(f"  Saved to: {result}")
    
    if args.analysis in ['stats', 'all']:
        print("\n=== Statistical Analysis ===")
        print("Generating descriptive statistics...")
        stats = statistical_analyzer.get_descriptive_statistics()
        
        # Print first few stats
        for stat in stats[:3]:
            print(f"  {stat}")
        
        print("Analyzing missing values...")
        missing = statistical_analyzer.get_missing_values_summary()
        for item in missing[:3]:
            print(f"  {item['column']}: {item['missing_values']} missing ({item['missing_percent']:.2f}%)")
        
        print("Analyzing outliers...")
        outliers = statistical_analyzer.get_outliers_summary()
        for item in outliers[:3]:
            print(f"  {item['column']}: {item['outlier_count']} outliers ({item['outlier_percent']:.2f}%)")
    
    if args.analysis in ['ml', 'all']:
        print("\n=== Machine Learning Analysis ===")
        numeric_columns = data_loader.get_numeric_columns()
        
        if len(numeric_columns) >= 2:
            print("Performing PCA analysis...")
            result = ml_analyzer.perform_pca(numeric_columns[:min(5, len(numeric_columns))], 2)
            print(f"  Explained variance: {result['explained_variance_ratio'][0]:.2f}, {result['explained_variance_ratio'][1]:.2f}")
            print(f"  Plots saved to: {result['plots']['variance']}, {result['plots']['scatter']}, {result['plots']['importance']}")
            
            print("Performing clustering analysis...")
            result = ml_analyzer.perform_clustering(numeric_columns[:min(5, len(numeric_columns))], 3)
            print(f"  Found {result['n_clusters']} clusters with sizes: {result['cluster_sizes']}")
            print(f"  Plots saved to: {result['plots']['scatter']}, {result['plots']['distribution']}")
        
        # If Gemini API key is provided, run AI analysis
        if args.gemini_key:
            print("\n=== Gemini AI Analysis ===")
            gemini_analyzer = GeminiAnalyzer(data_loader, output_dir=os.path.join(args.output_dir, 'gemini_outputs'))
            
            success = gemini_analyzer.configure(args.gemini_key)
            if success:
                print("Running AI analysis...")
                result = gemini_analyzer.analyze_data("Data Summary and Insights")
                if result:
                    print(f"  AI analysis saved to: {result['file_paths']['text']}")
            else:
                print("  Failed to configure Gemini API with provided key")
    
    print(f"\nAll analysis results saved to {args.output_dir} directory")

def run_batch_mode(args):
    """Run the application in batch mode to process multiple files."""
    if not args.input_dir:
        print("Error: --input-dir is required for batch mode")
        sys.exit(1)
    
    if not args.analysis:
        print("Error: --analysis type is required for batch mode")
        sys.exit(1)
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory")
        sys.exit(1)
    
    # Find all CSV files in the input directory
    csv_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print(f"Error: No CSV files found in {args.input_dir}")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV files in {args.input_dir}")
    
    # Process each file
    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(args.input_dir, csv_file)
        file_output_dir = os.path.join(args.output_dir, os.path.splitext(csv_file)[0])
        os.makedirs(file_output_dir, exist_ok=True)
        
        print(f"\n[{i+1}/{len(csv_files)}] Processing {csv_file}...")
        
        # Create a new args object with the current file
        file_args = argparse.Namespace(
            input=file_path,
            output_dir=file_output_dir,
            analysis=args.analysis,
            gemini_key=args.gemini_key
        )
        
        # Run CLI mode for this file
        run_cli_mode(file_args)
    
    print(f"\nBatch processing complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
