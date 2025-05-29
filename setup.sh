#!/bin/bash
# SpotLight Setup Script

echo "🔦 SpotLight Setup"
echo "=================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Found Python $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start using SpotLight:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the web app: python run_webapp.py"
echo "3. Or run the CLI: python run_cli.py"
echo ""
echo "🌐 Web interface will be available at: http://localhost:8080"