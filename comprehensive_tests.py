import unittest
import os
import pandas as pd
import json
import tempfile
import shutil
import datetime
import sys
import app as app_module
from app import app, load_and_analyze_data, generate_matplotlib_chart, generate_plotly_charts

class ComprehensiveTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests"""
        # Create sample data file for testing
        cls.temp_dir = tempfile.mkdtemp()
        cls.sample_csv_path = os.path.join(cls.temp_dir, "sample_azure_usage.csv")
        
        # Create a sample CSV with known data
        with open(cls.sample_csv_path, 'w') as f:
            f.write("Date,ServiceName,Cost,ResourceGroup,ResourceLocation\n")
            # Add a few weeks of data with known patterns
            for week in range(4):
                base_date = datetime.datetime(2023, 1, 1) + datetime.timedelta(days=week*7)
                date_str = base_date.strftime("%Y-%m-%d")
                
                # Service A is always the most expensive
                f.write(f"{date_str},Service A,{150.0 + week*10},ResourceGroup1,East US\n")
                # Service B is always second
                f.write(f"{date_str},Service B,{100.0 + week*5},ResourceGroup1,East US\n")
                # Service C and D flip between third and fourth place
                if week % 2 == 0:
                    f.write(f"{date_str},Service C,{80.0 + week*3},ResourceGroup2,West US\n")
                    f.write(f"{date_str},Service D,{70.0 + week*2},ResourceGroup2,West US\n")
                else:
                    f.write(f"{date_str},Service D,{85.0 + week*3},ResourceGroup2,West US\n")
                    f.write(f"{date_str},Service C,{65.0 + week*2},ResourceGroup2,West US\n")
                # Service E is always fifth
                f.write(f"{date_str},Service E,{50.0 + week},ResourceGroup3,Central US\n")
                # Service F is never in top 5
                f.write(f"{date_str},Service F,{40.0},ResourceGroup3,Central US\n")
        
        # Store the original CSV_FILE path
        cls.original_csv_file = app_module.CSV_FILE
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Restore the original CSV_FILE path
        app_module.CSV_FILE = cls.original_csv_file
        
        # Clean up temporary directory
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """Set up before each test"""
        # Use the sample data for testing
        app_module.CSV_FILE = self.sample_csv_path
        
    def test_data_processing_pipeline(self):
        """Test for the full data processing and visualization pipeline"""
        # 1. Load and analyze the data
        original_df, analysis_df = load_and_analyze_data()
        
        # 2. Verify the data analysis results
        self.assertIsInstance(original_df, pd.DataFrame)
        self.assertIsInstance(analysis_df, pd.DataFrame)
        
        # Check we have 4 weeks of data
        self.assertEqual(original_df['Week'].nunique(), 4)
        self.assertEqual(analysis_df['Week'].nunique(), 4)
        
        # Check that each week has exactly 5 top services in the analysis
        week_counts = analysis_df.groupby('Week').size()
        for week, count in week_counts.items():
            self.assertEqual(count, 5, f"Week {week} should have exactly 5 services")
            
        # Check that Service F is never in the top 5
        self.assertNotIn('Service F', analysis_df['ServiceName'].unique())
        
        # 3. Generate the charts
        chart_base64 = generate_matplotlib_chart(analysis_df)
        self.assertIsInstance(chart_base64, str)
        self.assertTrue(len(chart_base64) > 0)
        
        # 4. Generate Plotly charts
        plotly_charts = generate_plotly_charts(original_df, analysis_df)
        self.assertEqual(len(plotly_charts), 4)
        
    def test_malformed_csv_handling(self):
        """Test handling of malformed CSV files with missing headers"""
        # Create a malformed CSV
        malformed_csv_path = os.path.join(self.temp_dir, "malformed_azure_usage.csv")
        with open(malformed_csv_path, 'w') as f:
            # Missing required headers
            f.write("InvalidDate,Service,Amount\n")
            f.write("2023-01-01,Service A,100.0\n")
        
        # Temporarily set the app's CSV_FILE to our malformed file
        old_csv_file = app_module.CSV_FILE
        app_module.CSV_FILE = malformed_csv_path
        
        try:
            # Load and analyze should still work, but return empty DataFrames for analysis
            original_df, analysis_df = load_and_analyze_data()
            
            # Check that DataFrames are still returned
            self.assertIsInstance(original_df, pd.DataFrame)
            self.assertIsInstance(analysis_df, pd.DataFrame)
            
            # The analysis dataframe should be empty since we can't match the expected columns
            self.assertTrue('Cost' not in original_df.columns or analysis_df.empty)
        finally:
            # Restore the original CSV_FILE path
            app_module.CSV_FILE = old_csv_file
    
    def test_corrupt_date_handling(self):
        """Test handling of CSV files with corrupt dates"""
        # Create a CSV with corrupt dates
        corrupt_dates_csv_path = os.path.join(self.temp_dir, "corrupt_dates_azure_usage.csv")
        with open(corrupt_dates_csv_path, 'w') as f:
            f.write("Date,ServiceName,Cost,ResourceGroup,ResourceLocation\n")
            f.write("2023-01-01,Service A,100.0,ResourceGroup1,East US\n")
            f.write("NOT-A-DATE,Service B,50.0,ResourceGroup1,East US\n")
            f.write("2023/13/45,Service C,75.0,ResourceGroup2,West US\n")
            f.write("2023-01-02,Service D,125.0,ResourceGroup2,West US\n")
        
        # Temporarily set the app's CSV_FILE to our corrupt dates file
        old_csv_file = app_module.CSV_FILE
        app_module.CSV_FILE = corrupt_dates_csv_path
        
        try:
            # Load and analyze - it should filter out the invalid dates
            original_df, analysis_df = load_and_analyze_data()
            
            # Check that we only have valid dates
            self.assertEqual(len(original_df), 2)  # Only the valid rows
            self.assertEqual(len(analysis_df), 2)  # Both valid services are in top 5
        finally:
            # Restore the original CSV_FILE path
            app_module.CSV_FILE = old_csv_file
    
    def test_file_not_found_handling(self):
        """Test handling of non-existent files"""
        # Temporarily set the app's CSV_FILE to a non-existent path
        old_csv_file = app_module.CSV_FILE
        app_module.CSV_FILE = os.path.join(self.temp_dir, "doesnotexist.csv")
        
        try:
            # Load and analyze should raise FileNotFoundError
            with self.assertRaises(FileNotFoundError):
                load_and_analyze_data()
        finally:
            # Restore the original CSV_FILE path
            app_module.CSV_FILE = old_csv_file

if __name__ == "__main__":
    unittest.main() 