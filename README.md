# AIAP 21 Technical Assessment

## Project Overview

This repository contains the solution for the AIAP 21 Technical Assessment.

## Project Structure

```
├── data/                   # Store gas_monitoring.db here (DO NOT upload)
├── src/                    # Python scripts go here
├── eda.ipynb              # Exploratory Data Analysis notebook
├── run.sh                 # Run script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- Git
- SQLite database file (`gas_monitoring.db`)

### Quick Setup (Recommended)

1. **Place the database file**: Put `gas_monitoring.db` in the `data/` directory
2. **Run the setup script**:
   ```bash
   python setup_environment.py
   ```
3. **Activate virtual environment**:

   ```bash
   # On Windows
   venv\Scripts\activate

   # On Unix/Linux/Mac
   source venv/bin/activate
   ```

### Manual Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Irf2an/aiap21-irfaan-bin-ahmad-777a.git
   cd aiap21-irfaan-bin-ahmad-777a
   ```

2. **Create virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Place database file**: Put `gas_monitoring.db` in the `data/` directory

## Solution Features

### Part 1: Comprehensive Data Analysis

- **Database Schema Analysis**: Automatic detection and analysis of table structures
- **Data Quality Assessment**: Missing values, duplicates, and data type validation
- **Statistical Analysis**: Descriptive statistics and correlation analysis
- **Outlier Analysis**: IQR and Z-score methods with box plot visualizations
- **Visualization**: Automated plotting for numerical and categorical data

### Part 2: Machine Learning Models

- **Logistic Regression**: Linear baseline model with interpretable coefficients
- **Random Forest**: Ensemble method handling non-linear relationships
- **XGBoost**: State-of-the-art gradient boosting for high accuracy
- **Feature Engineering**: Sensor interactions, environmental indices, time-based features
- **Hyperparameter Tuning**: Grid search optimization for all models

### Key Components

- **`src/main.py`**: Machine learning pipeline with database validation
- **`src/models.py`**: Machine learning models and preprocessing pipeline
- **`eda.ipynb`**: Comprehensive exploratory data analysis notebook
- **`setup_environment.py`**: Automated environment setup

### Technical Features

- **Error Handling**: Robust error handling and validation
- **Memory Optimization**: Efficient data loading and processing
- **Modular Design**: Well-organized, reusable code components
- **Data Preprocessing**: KNN imputation, outlier handling, feature engineering
- **Model Evaluation**: Cross-validation, performance metrics, feature importance

## Usage

### Quick Start

1. Place the `gas_monitoring.db` file in the `data/` directory
2. Run the solution using the provided script:

   ```bash
   # On Unix/Linux/Mac
   chmod +x run.sh
   ./run.sh

   # On Windows
   run.bat
   ```

### Manual Execution

1. **Run main analysis**:

   ```bash
   python src/main.py
   ```

2. **Open Jupyter notebook**:

   ```bash
   jupyter notebook eda.ipynb
   ```

3. **Open Jupyter notebook for detailed EDA**:
   ```bash
   jupyter notebook eda.ipynb
   ```

### Expected Output

**Main Script (`src/main.py`):**

- Database connection validation
- Data availability check
- Machine learning pipeline execution
- Model training and performance comparison
- Feature importance analysis
- Cross-validation results

**Jupyter Notebook (`eda.ipynb`):**

- Comprehensive exploratory data analysis
- Database schema analysis
- Data quality assessment with missing data analysis
- Statistical summaries and distribution analysis
- Outlier detection with box plots
- Visualization plots and correlation analysis
- Detailed insights and recommendations

## Requirements

### Technical Requirements

- **Python 3.8+**: Required for modern Python features
- **SQLite Database**: `gas_monitoring.db` file in data directory
- **Memory**: Minimum 4GB RAM for data processing
- **Storage**: At least 1GB free space for virtual environment

### Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **sqlite3**: Database connectivity (built-in)
- **jupyter**: Interactive notebook environment
- **scikit-learn**: Machine learning utilities and preprocessing
- **xgboost**: Gradient boosting machine learning algorithm

### Data Requirements

- **Database Format**: SQLite (.db) file
- **Location**: Must be placed in `data/` directory
- **Access**: Read-only access required
- **Size**: No specific size limit (optimized for large datasets)

## Submission

### Repository Structure

```
├── data/                   # gas_monitoring.db (DO NOT upload)
├── src/                    # Python source code
│   ├── main.py            # Complete analysis pipeline (EDA + ML)
│   └── models.py          # Machine learning models and pipeline
├── eda.ipynb              # Exploratory data analysis with conclusions
├── run.sh                 # Execution script
├── setup_environment.py   # Environment setup
├── requirements.txt       # Dependencies
└── README.md             # Complete documentation
```

### Key Deliverables

1. **Part 1 - EDA**: Comprehensive exploratory data analysis with outlier detection
2. **Part 2 - ML**: Three trained machine learning models with performance comparison
3. **Source Code**: Complete Python implementation with modular design
4. **Analysis Notebook**: Interactive Jupyter notebook with detailed conclusions
5. **Documentation**: Comprehensive README with setup and usage instructions
6. **Setup Scripts**: Automated environment configuration

### Model Selection Rationale

**Why These Three Models?**

1. **Logistic Regression**:

   - Provides interpretable baseline performance
   - Fast training and prediction
   - Good for understanding linear relationships between sensors and activity
   - Works well with scaled features

2. **Random Forest**:

   - Handles non-linear relationships between sensors
   - Robust to outliers and missing values (important for this dataset)
   - Provides feature importance ranking
   - Good ensemble method for mixed data types

3. **XGBoost**:
   - State-of-the-art gradient boosting
   - Excellent performance on tabular data
   - Built-in regularization prevents overfitting
   - Handles missing values automatically
   - Often achieves highest accuracy

**Alternative Models Considered:**

- **SVM**: Could work but less interpretable and slower
- **Neural Networks**: Overkill for this dataset size, requires more data
- **Naive Bayes**: Too simple for complex sensor relationships
