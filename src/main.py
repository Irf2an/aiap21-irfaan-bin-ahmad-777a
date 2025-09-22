"""
Main entry point for the AIAP 21 Technical Assessment solution.
Gas Monitoring Data Analysis
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import ML pipeline
try:
    from .models import GasMonitoringMLPipeline
    ML_AVAILABLE = True
except ImportError:
    try:
        from models import GasMonitoringMLPipeline
        ML_AVAILABLE = True
    except ImportError:
        ML_AVAILABLE = False
        print("Machine learning modules not available. Install scikit-learn and xgboost.")

def check_database_exists():
    """Check if the gas_monitoring.db file exists in the data directory."""
    db_path = 'data/gas_monitoring.db'
    if not os.path.exists(db_path):
        print(f"Database file not found at {db_path}")
        print("Please place the gas_monitoring.db file in the data/ directory")
        return False
    print(f"Database file found at {db_path}")
    return True

def connect_to_database():
    """Connect to the SQLite database and return connection object."""
    try:
        conn = sqlite3.connect('data/gas_monitoring.db')
        print("Successfully connected to database")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def check_database_content(conn):
    """Check if database has data and return basic info."""
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    if not tables:
        print("No tables found in the database")
        return None, None
    
    # Get the main table (assuming first table is the primary one)
    main_table = tables[0][0]
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {main_table};")
    count = cursor.fetchone()[0]
    
    print(f"Database contains {count:,} records in table '{main_table}'")
    
    if count == 0:
        print("Warning: Database is empty!")
        return None, None
    
    return main_table, count

def main():
    """
    Main function to run the gas monitoring machine learning pipeline.
    """
    print("AIAP 21 Technical Assessment - Gas Monitoring ML Pipeline")
    print("=" * 60)
    
    # Check if database exists
    if not check_database_exists():
        return
    
    # Connect to database
    conn = connect_to_database()
    if conn is None:
        return
    
    try:
        # Check database content
        main_table, record_count = check_database_content(conn)
        if main_table is None:
            return
        
        # Machine Learning Analysis
        if ML_AVAILABLE:
            print("\nMACHINE LEARNING ANALYSIS")
            print("=" * 50)
            
            # Load data for ML
            query = f"SELECT * FROM {main_table}"
            df = pd.read_sql_query(query, conn)
            
            # Initialize ML pipeline
            pipeline = GasMonitoringMLPipeline(target_column='Activity Level')
            
            # Preprocess data
            processed_data = pipeline.load_and_preprocess_data(df)
            
            # Prepare features and target
            pipeline.prepare_features_and_target(processed_data)
            
            # Train all models
            ml_results = pipeline.train_all_models()
            
            # Display model explanations
            explanations = pipeline.get_model_explanations()
            print("\nMODEL SELECTION EXPLANATIONS")
            print("=" * 50)
            for model_name, explanation in explanations.items():
                print(f"\n{model_name}:")
                print("Why chosen:")
                for reason in explanation['why_chosen']:
                    print(f"  • {reason}")
            
            print("\n✅ Machine learning analysis completed successfully!")
            print("\n📊 For detailed exploratory data analysis, open eda.ipynb in Jupyter:")
            print("   jupyter notebook eda.ipynb")
        else:
            print("\n❌ Machine learning modules not available.")
            print("Please install required packages: pip install scikit-learn xgboost")
            print("\n📊 For detailed exploratory data analysis, open eda.ipynb in Jupyter:")
            print("   jupyter notebook eda.ipynb")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
    finally:
        conn.close()
        print("\nDatabase connection closed")

if __name__ == "__main__":
    main()
