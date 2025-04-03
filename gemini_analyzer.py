import streamlit as st
import google.generativeai as genai
import io
import pandas as pd

class GeminiAnalyzer:
    """
    A class for integrating Google's Gemini AI for data analysis.
    """
    
    def __init__(self, data_loader):
        """
        Initialize the GeminiAnalyzer class.
        
        Args:
            data_loader: An instance of the DataLoader class
        """
        self.data_loader = data_loader
        
        # Initialize session state variables if they don't exist
        if 'gemini_api_key' not in st.session_state:
            st.session_state.gemini_api_key = ""
        if 'gemini_configured' not in st.session_state:
            st.session_state.gemini_configured = False
    
    def create_api_key_input(self):
        """
        Create an input widget for the Gemini API key.
        
        Returns:
            bool: True if API key is configured successfully, False otherwise
        """
        api_key = st.text_input("Enter Gemini API Key", type="password", value=st.session_state.gemini_api_key)
        
        if api_key:
            st.session_state.gemini_api_key = api_key
            try:
                genai.configure(api_key=api_key)
                st.session_state.gemini_configured = True
                st.success("Gemini API configured successfully!")
                return True
            except Exception as e:
                st.error(f"Error configuring Gemini API: {e}")
                st.session_state.gemini_configured = False
                return False
        
        return st.session_state.gemini_configured
    
    def is_configured(self):
        """
        Check if Gemini API is configured.
        
        Returns:
            bool: True if configured, False otherwise
        """
        return st.session_state.gemini_configured
    
    def prepare_data_description(self):
        """
        Prepare a description of the data for Gemini.
        
        Returns:
            str: A string containing data description
        """
        df = self.data_loader.get_data()
        
        if df is not None:
            # Get data info
            buffer = io.StringIO()
            df.info(buf=buffer)
            data_info = buffer.getvalue()
            
            # Get data sample
            data_sample = df.head(5).to_string()
            
            # Get data statistics
            data_stats = df.describe().to_string()
            
            return {
                'info': data_info,
                'sample': data_sample,
                'stats': data_stats
            }
        
        return None
    
    def generate_prompt(self, analysis_type, custom_question=None):
        """
        Generate a prompt for Gemini based on the analysis type.
        
        Args:
            analysis_type (str): Type of analysis to perform
            custom_question (str, optional): Custom question for analysis
            
        Returns:
            str: Generated prompt
        """
        data_desc = self.prepare_data_description()
        
        if data_desc is None:
            return None
        
        if analysis_type == "Data Summary and Insights":
            prompt = f"""
            I have a dataset with the following structure:
            
            {data_desc['info']}
            
            Here's a sample of the data:
            {data_desc['sample']}
            
            And here are the basic statistics:
            {data_desc['stats']}
            
            Please provide a comprehensive summary of this dataset, including:
            1. The main characteristics and patterns in the data
            2. Key insights about the variables and their distributions
            3. Potential business or analytical value of this dataset
            4. Suggestions for further analysis or data collection
            """
        
        elif analysis_type == "Correlation Analysis":
            # Include correlation matrix
            df = self.data_loader.get_data()
            corr_matrix = df.select_dtypes(include=['number']).corr().to_string()
            
            prompt = f"""
            I have a dataset with the following structure:
            
            {data_desc['info']}
            
            Here's the correlation matrix of numeric variables:
            {corr_matrix}
            
            Please analyze these correlations and provide:
            1. The most significant positive and negative correlations
            2. Explanation of what these correlations might mean
            3. Potential causal relationships to investigate
            4. Variables that show little correlation with others and what that implies
            """
        
        elif analysis_type == "Trend Identification":
            prompt = f"""
            I have a dataset with the following structure:
            
            {data_desc['info']}
            
            Here's a sample of the data:
            {data_desc['sample']}
            
            And here are the basic statistics:
            {data_desc['stats']}
            
            Please identify potential trends in this data, including:
            1. Any apparent patterns or trends across variables
            2. Groups or segments that might exist in the data
            3. How these trends might evolve based on the data
            4. Recommendations for visualizing or further analyzing these trends
            """
        
        elif analysis_type == "Anomaly Detection":
            prompt = f"""
            I have a dataset with the following structure:
            
            {data_desc['info']}
            
            Here's a sample of the data:
            {data_desc['sample']}
            
            And here are the basic statistics:
            {data_desc['stats']}
            
            Please help identify potential anomalies or outliers in this data:
            1. Which variables might contain outliers based on the statistics
            2. What might be causing these anomalies
            3. How these outliers might affect analysis results
            4. Recommendations for handling these anomalies
            """
        
        else:  # Custom Analysis
            if custom_question is None:
                custom_question = "What are the most interesting insights from this dataset and what actions would you recommend based on them?"
            
            prompt = f"""
            I have a dataset with the following structure:
            
            {data_desc['info']}
            
            Here's a sample of the data:
            {data_desc['sample']}
            
            And here are the basic statistics:
            {data_desc['stats']}
            
            Please answer the following question about this data:
            {custom_question}
            """
        
        return prompt
    
    def analyze_data(self, analysis_type, custom_question=None):
        """
        Analyze data using Gemini AI.
        
        Args:
            analysis_type (str): Type of analysis to perform
            custom_question (str, optional): Custom question for analysis
            
        Returns:
            str: Analysis result from Gemini
        """
        if not self.is_configured():
            st.warning("Please configure your Gemini API key to use AI analysis.")
            return None
        
        prompt = self.generate_prompt(analysis_type, custom_question)
        
        if prompt is None:
            st.error("Failed to generate prompt. Please make sure data is loaded.")
            return None
        
        try:
            # Configure Gemini model
            model = genai.GenerativeModel('gemini-2.0-flash')
            st.write(f"PASSING PROMPT: {prompt}")
            
            # Generate response

            response = model.generate_content(prompt)
            st.write(f"MODEL OUTPUT: {response.candidates[0].content.parts[0].text}")

            return response.candidates[0].content.parts[0].text
        
        except Exception as e:
            st.error(f"Error generating AI analysis: {e}")
            st.info("Please check your API key and try again.")
            return None
    
    def get_analysis_types(self):
        """
        Get available analysis types.
        
        Returns:
            list: List of available analysis types
        """
        return [
            "Data Summary and Insights",
            "Correlation Analysis",
            "Trend Identification",
            "Anomaly Detection",
            "Custom Analysis"
        ]
