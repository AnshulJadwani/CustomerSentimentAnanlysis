#!/bin/bash

# Customer Sentiment Analysis - Setup Script
# This script automates the installation and setup process

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════════"
echo "  Customer Sentiment Analysis - Setup Script"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Function to print colored output
print_success() {
    echo -e "\033[0;32m✓ $1\033[0m"
}

print_error() {
    echo -e "\033[0;31m✗ $1\033[0m"
}

print_info() {
    echo -e "\033[0;34mℹ $1\033[0m"
}

print_warning() {
    echo -e "\033[0;33m⚠ $1\033[0m"
}

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ -d "venv" ]; then
    print_warning "Virtual environment already exists. Skipping creation."
else
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Install requirements
print_info "Installing Python packages (this may take a few minutes)..."
if pip install -r requirements.txt > /dev/null 2>&1; then
    print_success "All packages installed successfully"
else
    print_error "Failed to install packages. Check requirements.txt"
    exit 1
fi

# Download NLTK data
print_info "Downloading NLTK data..."
python3 << END
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
print("✓ NLTK data downloaded")
END
print_success "NLTK data downloaded"

# Check if data file exists
if [ -f "data/clean_review.csv" ]; then
    print_success "Dataset found: data/clean_review.csv"
else
    print_warning "Dataset not found. Please add clean_review.csv to data/ directory"
fi

# Create models directory if it doesn't exist
mkdir -p models
print_success "Models directory ready"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Setup Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo ""
echo "1. Train models:"
echo "   python main.py train --data data/clean_review.csv"
echo ""
echo "2. Run Jupyter notebook:"
echo "   jupyter notebook notebooks/sentiment_analysis.ipynb"
echo ""
echo "3. Launch web app:"
echo "   streamlit run app/app.py"
echo ""
echo "4. Make a prediction:"
echo "   python main.py predict --text 'Your review text here'"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📚 Documentation:"
echo "   - README.md      - Full documentation"
echo "   - QUICKSTART.md  - Quick start guide"
echo "   - WORKFLOW.md    - Pipeline workflow"
echo ""
echo "For help, visit: https://github.com/yourusername/CustomerSentimentAnalysis"
echo ""
