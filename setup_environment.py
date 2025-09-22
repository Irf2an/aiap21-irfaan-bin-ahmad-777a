"""
Environment setup script for AIAP 21 Technical Assessment.
This script sets up the development environment and installs dependencies.
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please install Python 3.8 or higher")
        return False

def create_virtual_environment():
    """Create virtual environment."""
    if os.path.exists("venv"):
        print("📁 Virtual environment already exists")
        return True
    
    return run_command("python -m venv venv", "Creating virtual environment")

def activate_virtual_environment():
    """Get the activation command for virtual environment."""
    if platform.system() == "Windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_dependencies():
    """Install required dependencies."""
    if platform.system() == "Windows":
        pip_command = "venv\\Scripts\\pip"
    else:
        pip_command = "venv/bin/pip"
    
    # Upgrade pip first
    run_command(f"{pip_command} install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    return run_command(f"{pip_command} install -r requirements.txt", "Installing dependencies")

def check_database_file():
    """Check if database file exists."""
    db_path = "data/gas_monitoring.db"
    if os.path.exists(db_path):
        print(f"✅ Database file found at {db_path}")
        return True
    else:
        print(f"⚠️  Database file not found at {db_path}")
        print("Please place the gas_monitoring.db file in the data/ directory")
        return False

def run_tests():
    """Run the test suite."""
    if platform.system() == "Windows":
        python_command = "venv\\Scripts\\python"
    else:
        python_command = "venv/bin/python"
    
    return run_command(f"{python_command} src/test_solution.py", "Running tests")

def main():
    """Main setup function."""
    print("🚀 AIAP 21 Technical Assessment - Environment Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    if not create_virtual_environment():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Check database file
    check_database_file()
    
    # Run tests
    print("\n🧪 Running tests...")
    run_tests()
    
    print("\n🎯 Setup Complete!")
    print("=" * 60)
    print("Next steps:")
    print("1. Place gas_monitoring.db in the data/ directory")
    print("2. Activate virtual environment:")
    print(f"   {activate_virtual_environment()}")
    print("3. Run the main solution:")
    print("   python src/main.py")
    print("4. Open eda.ipynb for detailed analysis")
    print("5. Use run.sh to run the complete solution")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
