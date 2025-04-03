import pandas as pd
import streamlit as st
import io
import base64

class DataLoader:
    """
    A class for loading, preprocessing, and managing CSV data.
    """
    
    def __init__(self):
        """Initialize the DataLoader class."""
        # Initialize session state variables if they don't exist
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'filename' not in st.session_state:
            st.session_state.filename = None
    
    def create_upload_widget(self):
        """
        Create a file uploader widget for CSV files.
        
        Returns:
            uploaded_file: The uploaded file object or None
        """
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        return uploaded_file
    
    def load_data(self, uploaded_file):
        """
        Load data from an uploaded CSV file.
        
        Args:
            uploaded_file: The uploaded file object
            
        Returns:
            bool: True if loading was successful, False otherwise
        """
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.session_state.filename = uploaded_file.name
                return True
            except Exception as e:
                st.error(f"Error: {e}")
                return False
        return False
    
    def get_data(self):
        """
        Get the currently loaded data.
        
        Returns:
            pandas.DataFrame: The loaded data or None if no data is loaded
        """
        return st.session_state.data
    
    def get_filename(self):
        """
        Get the filename of the currently loaded data.
        
        Returns:
            str: The filename or None if no file is loaded
        """
        return st.session_state.filename
    
    def get_data_info(self):
        """
        Get basic information about the loaded data.
        
        Returns:
            dict: A dictionary containing basic information about the data
        """
        if st.session_state.data is not None:
            df = st.session_state.data
            
            # Calculate memory usage
            memory_usage = df.memory_usage(deep=True).sum()
            if memory_usage < 1024:
                memory_str = f"{memory_usage} bytes"
            elif memory_usage < 1024**2:
                memory_str = f"{memory_usage/1024:.2f} KB"
            else:
                memory_str = f"{memory_usage/(1024**2):.2f} MB"
            
            # Count missing values
            missing_values = df.isnull().sum().sum()
            
            return {
                "filename": st.session_state.filename,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "memory_usage": memory_str,
                "missing_values": missing_values
            }
        return None
    
    def get_column_info(self):
        """
        Get detailed information about each column in the data.
        
        Returns:
            list: A list of dictionaries containing information about each column
        """
        if st.session_state.data is not None:
            df = st.session_state.data
            
            col_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                unique = df[col].nunique()
                missing = df[col].isnull().sum()
                missing_pct = (missing / len(df)) * 100
                
                col_info.append({
                    "Column": col,
                    "Data Type": dtype,
                    "Unique Values": unique,
                    "Missing Values": missing,
                    "Missing (%)": f"{missing_pct:.2f}%"
                })
            
            return col_info
        return None
    
    def get_numeric_columns(self):
        """
        Get a list of numeric columns in the data.
        
        Returns:
            list: A list of numeric column names
        """
        if st.session_state.data is not None:
            return st.session_state.data.select_dtypes(include=['number']).columns.tolist()
        return []
    
    def get_categorical_columns(self):
        """
        Get a list of categorical columns in the data.
        
        Returns:
            list: A list of categorical column names
        """
        if st.session_state.data is not None:
            return st.session_state.data.select_dtypes(include=['object', 'category']).columns.tolist()
        return []
    
    def create_download_link(self, processed_df=None):
        """
        Create a download link for the data.
        
        Args:
            processed_df (pandas.DataFrame, optional): A processed dataframe to download.
                If None, the original data will be used.
                
        Returns:
            str: HTML link for downloading the data
        """
        if processed_df is None and st.session_state.data is not None:
            processed_df = st.session_state.data
        
        if processed_df is not None and st.session_state.filename is not None:
            csv = processed_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            filename = st.session_state.filename.split(".")[0]
            return f'<a href="data:file/csv;base64,{b64}" download="{filename}_processed.csv">Download CSV File</a>'
        return None
