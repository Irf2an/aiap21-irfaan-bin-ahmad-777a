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

def check_database_exists():
    """Check if the gas_monitoring.db file exists in the data directory."""
    db_path = 'data/gas_monitoring.db'
    if not os.path.exists(db_path):
        print(f"❌ Database file not found at {db_path}")
        print("Please place the gas_monitoring.db file in the data/ directory")
        return False
    print(f"✅ Database file found at {db_path}")
    return True

def connect_to_database():
    """Connect to the SQLite database and return connection object."""
    try:
        conn = sqlite3.connect('data/gas_monitoring.db')
        print("✅ Successfully connected to database")
        return conn
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return None

def explore_database_schema(conn):
    """Explore the database schema to understand table structure."""
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("\n📊 Database Schema Analysis")
    print("=" * 40)
    print(f"Found {len(tables)} tables:")
    
    for table in tables:
        table_name = table[0]
        print(f"\n📋 Table: {table_name}")
        
        # Get column information
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        print("Columns:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        count = cursor.fetchone()[0]
        print(f"  Row count: {count:,}")
    
    return tables

def load_sample_data(conn, table_name, limit=5):
    """Load a sample of data from a table."""
    try:
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        print(f"\n📄 Sample data from {table_name}:")
        print(df.to_string(index=False))
        return df
    except Exception as e:
        print(f"❌ Error loading data from {table_name}: {e}")
        return None

def generate_basic_statistics(conn, table_name):
    """Generate basic statistics for a table."""
    try:
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        
        print(f"\n📈 Basic Statistics for {table_name}")
        print("=" * 50)
        print(f"Shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nData Types:")
        print(df.dtypes)
        
        print("\nMissing Values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values found")
        
        print("\nDescriptive Statistics:")
        print(df.describe())
        
        return df
    except Exception as e:
        print(f"❌ Error generating statistics for {table_name}: {e}")
        return None

def main():
    """
    Main function to run the gas monitoring data analysis.
    """
    print("🔬 AIAP 21 Technical Assessment - Gas Monitoring Data Analysis")
    print("=" * 70)
    
    # Check if database exists
    if not check_database_exists():
        return
    
    # Connect to database
    conn = connect_to_database()
    if conn is None:
        return
    
    try:
        # Explore database schema
        tables = explore_database_schema(conn)
        
        if not tables:
            print("❌ No tables found in the database")
            return
        
        # Analyze each table
        for table in tables:
            table_name = table[0]
            print(f"\n🔍 Analyzing table: {table_name}")
            print("-" * 50)
            
            # Load sample data
            sample_df = load_sample_data(conn, table_name)
            
            # Generate statistics
            full_df = generate_basic_statistics(conn, table_name)
            
            print(f"\n✅ Analysis completed for {table_name}")
        
        print("\n🎯 Next Steps:")
        print("1. Open eda.ipynb for detailed exploratory data analysis")
        print("2. Review the generated statistics and data samples")
        print("3. Identify key patterns and insights in the data")
        print("4. Create visualizations and further analysis")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
    finally:
        conn.close()
        print("\n🔒 Database connection closed")

if __name__ == "__main__":
    main()
