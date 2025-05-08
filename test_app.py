import unittest
import os
import pandas as pd
import json
import tempfile
import shutil
from datetime import datetime
import sys
import app as app_module
from app import app, load_and_analyze_data, generate_matplotlib_chart, generate_plotly_charts

class AzureUsageAnalysisTests(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Use the sample data for testing
        self.csv_path = os.path.join("data", "AzureUsage.csv")
        self.assertTrue(os.path.exists(self.csv_path), f"CSV file not found at {self.csv_path}")
        
        # Store the original CSV_FILE path
        self.original_csv_file = app_module.CSV_FILE
        
        # Point the app to our test CSV file
        app_module.CSV_FILE = self.csv_path
    
    def tearDown(self):
        """Clean up after tests"""
        # Restore the original CSV_FILE path
        app_module.CSV_FILE = self.original_csv_file
    
    def test_index_route(self):
        """Test the main index route"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Azure Usage Analysis Dashboard', response.data)
        self.assertIn(b'Weekly Cost Trend', response.data)
        self.assertIn(b'Top 5 Services', response.data)
        self.assertIn(b'Service Cost Distribution', response.data)
        self.assertIn(b'Service Comparison by Week', response.data)
        self.assertIn(b'Weekly Top 5 Cost Breakdown', response.data)
        
    def test_api_data_route(self):
        """Test the API data route"""
        response = self.app.get('/api/data')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # API now returns a dictionary with chart data and statistics
        self.assertIsInstance(data, dict)
        
        # Check for key fields in the response
        self.assertIn('weekly_trend_json', data)
        self.assertIn('service_pie_json', data)
        self.assertIn('service_comparison_json', data)
        self.assertIn('service_multiline_json', data)
        self.assertIn('weekly_breakdown_json', data)
        self.assertIn('total_cost', data)
        self.assertIn('avg_weekly_cost', data)
        self.assertIn('top_service', data)
        
        # Ensure total cost is a valid number
        self.assertTrue(float(data['total_cost']) >= 0)
        
        # Verify specific data from our actual file
        # Virtual Machines, Storage, Application Gateway, or Azure DDOS Protection should be in top services
        common_services = ['Virtual Machines', 'Storage', 'Application Gateway', 'Azure DDOS Protection']
        self.assertTrue(any(service in data['top_service'] for service in common_services), 
                        f"Expected one of {common_services} to be a top service, but got {data['top_service']}")
    
    def test_data_loading(self):
        """Test data loading and analysis functions"""
        original_df, analysis_df = load_and_analyze_data()
        
        # Check that DataFrames are returned
        self.assertIsInstance(original_df, pd.DataFrame)
        self.assertIsInstance(analysis_df, pd.DataFrame)
        
        # Check original data has expected columns
        self.assertIn('Date', original_df.columns)
        self.assertIn('ServiceName', original_df.columns)
        self.assertIn('Cost', original_df.columns)
        self.assertIn('Week', original_df.columns)
        
        # Check analysis data structure
        self.assertIn('Week', analysis_df.columns)
        self.assertIn('ServiceName', analysis_df.columns)
        self.assertIn('Cost', analysis_df.columns)
        
        # Check that we have some unique weeks of data
        unique_weeks = original_df['Week'].nunique()
        self.assertTrue(unique_weeks > 0, "Expected at least one unique week in the data")
        
        # Check that the analysis contains data for each week
        weeks_in_analysis = analysis_df['Week'].nunique()
        self.assertEqual(weeks_in_analysis, unique_weeks, "Analysis should contain data for all weeks")
        
        # Check that there are at most 5 rows per week in the analysis (top 5)
        week_counts = analysis_df.groupby('Week').size()
        for week, count in week_counts.items():
            self.assertLessEqual(count, 5, f"Expected at most 5 services for week {week}")
    
    def test_chart_generation(self):
        """Test chart generation functions"""
        _, analysis_df = load_and_analyze_data()
        
        # Test matplotlib chart generation
        chart_base64 = generate_matplotlib_chart(analysis_df)
        # Check that we get a non-empty base64 string
        self.assertIsInstance(chart_base64, str)
        self.assertTrue(len(chart_base64) > 0)
        
        # Test plotly chart generation
        original_df, analysis_df = load_and_analyze_data()
        charts = generate_plotly_charts(original_df, analysis_df)
        
        # Check that all expected charts are present
        self.assertIn('weekly_trend', charts)
        self.assertIn('service_pie', charts)
        self.assertIn('service_comparison', charts)
        
        # Check chart data format
        for chart_name, chart_data in charts.items():
            # Ensure it's a JSON string
            self.assertIsInstance(chart_data, str)
            # Ensure it can be parsed as JSON
            try:
                parsed_chart = json.loads(chart_data)
                self.assertIsInstance(parsed_chart, dict)
                # Check for key properties in Plotly charts
                self.assertIn('data', parsed_chart)
                self.assertIn('layout', parsed_chart)
            except json.JSONDecodeError:
                self.fail(f"Chart {chart_name} does not contain valid JSON")
    
    def test_missing_data_handling(self):
        """Test handling of files with missing or malformed data"""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file with some invalid dates
            test_csv_path = os.path.join(temp_dir, "test_azure_usage.csv")
            with open(test_csv_path, 'w') as f:
                f.write("Date,ServiceName,Cost,ResourceGroup,ResourceLocation\n")
                f.write("2023-01-01,Virtual Machines,125.45,RG-Production,East US\n")
                f.write("INVALID-DATE,Storage,45.20,RG-Production,East US\n")  # Invalid date
                f.write("2023-01-01,Logic Apps,89.75,RG-Database,West US\n")
            
            # Temporarily set the app's CSV_FILE to our test file
            old_csv_file = app_module.CSV_FILE
            app_module.CSV_FILE = test_csv_path
            
            try:
                # Test that the data loading function handles the invalid date
                original_df, analysis_df = load_and_analyze_data()
                
                # Check that DataFrames are still returned
                self.assertIsInstance(original_df, pd.DataFrame)
                self.assertIsInstance(analysis_df, pd.DataFrame)
                
                # Verify there are only 2 valid rows with dates (the invalid one should be NaT)
                valid_dates = original_df['Date'].dropna()
                self.assertEqual(len(valid_dates), 2)
                
                # Check that the week values are properly calculated for valid dates
                weeks = original_df['Week'].dropna()
                self.assertEqual(len(weeks), 2)
            finally:
                # Restore the original CSV_FILE path
                app_module.CSV_FILE = old_csv_file
    
    def test_empty_data_handling(self):
        """Test handling of empty data file"""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create an empty test file with just headers
            test_csv_path = os.path.join(temp_dir, "empty_azure_usage.csv")
            with open(test_csv_path, 'w') as f:
                f.write("Date,ServiceName,Cost,ResourceGroup,ResourceLocation\n")
            
            # Temporarily set the app's CSV_FILE to our test file
            old_csv_file = app_module.CSV_FILE
            app_module.CSV_FILE = test_csv_path
            
            try:
                # Test that the data loading function handles the empty file
                original_df, analysis_df = load_and_analyze_data()
                
                # Check that DataFrames are still returned
                self.assertIsInstance(original_df, pd.DataFrame)
                self.assertIsInstance(analysis_df, pd.DataFrame)
                
                # Verify the analysis DataFrame is empty
                self.assertEqual(len(analysis_df), 0)
            finally:
                # Restore the original CSV_FILE path
                app_module.CSV_FILE = old_csv_file

if __name__ == '__main__':
    unittest.main() 