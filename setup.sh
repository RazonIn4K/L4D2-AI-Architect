#!/bin/bash
# L4D2-AI-Architect Complete Setup Script
# 
# This script sets up the entire project including:
# - Python environment
# - Dependencies
# - Game server configuration
# - Initial data collection

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}=====================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=====================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get OS type
OS="linux"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    OS="windows"
fi

print_header "L4D2-AI-Architect Setup"

# Check Python version
print_header "Checking Python"
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$PYTHON_VERSION >= 3.10" | bc -l) -eq 1 ]]; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.10+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check CUDA (optional but recommended)
print_header "Checking CUDA"
if command_exists nvidia-smi; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    print_success "CUDA $CUDA_VERSION detected"
    GPU_AVAILABLE=true
else
    print_warning "CUDA not detected. Training will use CPU (slower)"
    GPU_AVAILABLE=false
fi

# Create virtual environment
print_header "Setting up Python Environment"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Created virtual environment"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Activated virtual environment"

# Upgrade pip
pip install --upgrade pip

# Install dependencies
print_header "Installing Dependencies"
pip install -r requirements.txt
print_success "Installed Python dependencies"

# Install Unsloth separately if GPU available
if [ "$GPU_AVAILABLE" = true ]; then
    print_header "Installing Unsloth (GPU optimized)"
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    print_success "Installed Unsloth"
fi

# Create necessary directories
print_header "Creating Directory Structure"
mkdir -p data/{raw,processed,training_logs}
mkdir -p data/l4d2_server/{addons/sourcemod/{plugins,scripting,configs},cfg,maps}
mkdir -p model_adapters
mkdir -p exports
mkdir -p logs

# Create .gitkeep files
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/training_logs/.gitkeep
touch model_adapters/.gitkeep
touch exports/.gitkeep
touch logs/.gitkeep

print_success "Created directory structure"

# Setup environment configuration
print_header "Environment Configuration"
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_warning "Please edit .env file with your configuration"
    print_warning "Especially set L4D2_INSTALL_PATH and SRCDS_PATH"
else
    print_success ".env file already exists"
fi

# Make scripts executable
chmod +x activate.sh
chmod +x run_scraping.sh
chmod +x run_training.sh
chmod +x scripts/inference/copilot_cli.py
chmod +x scripts/director/director.py

print_success "Made scripts executable"

# Install SourceMod (placeholder instructions)
print_header "SourceMod Setup"
echo "To complete the setup:"
echo "1. Download SourceMod 1.11 from: https://www.sourcemod.net/downloads.php"
echo "2. Extract to: data/l4d2_server/addons/sourcemod/"
echo "3. Download Metamod: Source from: https://www.sourcemm.net/downloads.php"
echo "4. Extract to: data/l4d2_server/addons/metamod/"
print_warning "SourceMod and Metamod must be installed manually"

# Compile SourceMod plugin
print_header "Compiling SourceMod Plugin"
if [ -f "data/l4d2_server/addons/sourcemod/scripting/spcomp" ]; then
    data/l4d2_server/addons/sourcemod/scripting/spcomp \
        data/l4d2_server/addons/sourcemod/scripting/l4d2_ai_bridge.sp \
        -o data/l4d2_server/addons/sourcemod/plugins/l4d2_ai_bridge.smx
    print_success "Compiled SourceMod plugin"
else
    print_warning "SourceMod compiler not found. Please compile l4d2_ai_bridge.sp manually"
fi

# Download base model (optional)
print_header "Base Model Setup"
read -p "Do you want to download the base model now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Note: This will download ~7GB of data"
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Using HuggingFace CLI if available
        if command_exists huggingface-cli; then
            huggingface-cli download unsloth/mistral-7b-instruct-v0.3-bnb-4bit \
                --local-dir models/base/mistral-7b-instruct-v0.3-bnb-4bit
            print_success "Downloaded base model"
        else
            print_warning "Please install huggingface-cli and run:"
            echo "huggingface-cli download unsloth/mistral-7b-instruct-v0.3-bnb-4bit \\"
            echo "  --local-dir models/base/mistral-7b-instruct-v0.3-bnb-4bit"
        fi
    fi
fi

# Create systemd service files (Linux only)
if [ "$OS" = "linux" ]; then
    print_header "Creating Service Files"
    
    # Director service
    cat > l4d2-director.service << EOF
[Unit]
Description=L4D2 AI Director
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python scripts/director/director.py --mode rule
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # Copilot service
    cat > l4d2-copilot.service << EOF
[Unit]
Description=L4D2 Copilot Inference Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python scripts/inference/copilot_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    print_success "Created systemd service files"
    print_warning "To install services, run:"
    echo "sudo cp l4d2-director.service /etc/systemd/system/"
    echo "sudo cp l4d2-copilot.service /etc/systemd/system/"
    echo "sudo systemctl daemon-reload"
    echo "sudo systemctl enable l4d2-director"
    echo "sudo systemctl start l4d2-director"
fi

# Final instructions
print_header "Setup Complete!"
echo
echo "Next steps:"
echo "1. Edit .env with your paths and configuration"
echo "2. Install SourceMod and Metamod in data/l4d2_server/"
echo "3. Set up your L4D2 dedicated server"
echo "4. Run initial data collection:"
echo "   ./run_scraping.sh"
echo "5. Train your model:"
echo "   ./run_training.sh"
echo "6. Start the services:"
echo "   - Director: python scripts/director/director.py"
echo "   - Copilot: python scripts/inference/copilot_server.py"
echo
echo "For detailed instructions, see docs/TRAINING_GUIDE.md"
echo
print_success "Setup completed successfully!"

# Check if user wants to run initial tests
read -p "Run initial component tests? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_header "Running Component Tests"
    
    # Test director bridge
    echo "Testing director bridge (mock mode)..."
    python scripts/director/bridge.py --mock --test &
    BRIDGE_PID=$!
    sleep 2
    kill $BRIDGE_PID 2>/dev/null
    print_success "Bridge test passed"
    
    # Test copilot CLI
    echo "Testing copilot CLI templates..."
    python scripts/inference/copilot_cli.py template plugin --output /tmp/test_plugin.sp
    if [ -f "/tmp/test_plugin.sp" ]; then
        print_success "Copilot CLI test passed"
        rm /tmp/test_plugin.sp
    fi
    
    # Test imports
    echo "Testing Python imports..."
    python -c "
import sys
sys.path.insert(0, 'scripts')
try:
    from director.director import L4D2Director
    from inference.copilot_server import CopilotServer
    print('✓ All imports successful')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"
    print_success "All tests passed!"
fi

echo
print_success "L4D2-AI-Architect is ready to use!"
