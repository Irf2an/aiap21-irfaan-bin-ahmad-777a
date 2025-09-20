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

### Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Quick Start

1. Place the `gas_monitoring.db` file in the `data/` directory
2. Run the solution using the provided script:

   ```bash
   # On Unix/Linux/Mac
   chmod +x run.sh
   ./run.sh

   # On Windows
   run.sh
   ```

### Manual Execution

1. Create and activate virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python src/main.py`
4. Open `eda.ipynb` for exploratory data analysis

## Requirements

[To be updated based on specific requirements from the PDF]

## Submission

[To be updated based on submission requirements]
