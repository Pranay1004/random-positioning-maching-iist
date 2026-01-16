#!/bin/bash
# =============================================================================
# RPM Digital Twin - Setup Script
# =============================================================================
# This script sets up the development environment for the RPM Digital Twin.
# Run this after cloning the repository.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# Requirements:
#   - Python 3.11+
#   - pip
#   - (Optional) MATLAB R2024a
#   - (Optional) Docker for databases
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print banner
echo -e "${BLUE}"
echo "======================================"
echo "  RPM Digital Twin - Setup Script"
echo "======================================"
echo -e "${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]); then
    echo -e "${RED}Error: Python 3.11+ is required. Found: $PYTHON_VERSION${NC}"
    echo "Please install Python 3.11 or later."
    exit 1
fi
echo -e "${GREEN}Python $PYTHON_VERSION found.${NC}"

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Install development dependencies
echo -e "${YELLOW}Installing development dependencies...${NC}"
pip install pytest pytest-asyncio pytest-cov black isort mypy

# Create necessary directories
echo -e "${YELLOW}Creating directory structure...${NC}"
mkdir -p logs
mkdir -p data/experiments
mkdir -p data/calibration
mkdir -p data/exports

# Check for MATLAB (optional)
echo -e "${YELLOW}Checking for MATLAB...${NC}"
if command -v matlab &> /dev/null; then
    MATLAB_VERSION=$(matlab -batch "disp(version)" 2>/dev/null | tail -1)
    echo -e "${GREEN}MATLAB found: $MATLAB_VERSION${NC}"
    
    # Try to install MATLAB Engine for Python
    echo -e "${YELLOW}Installing MATLAB Engine for Python...${NC}"
    MATLAB_ROOT=$(matlab -batch "disp(matlabroot)" 2>/dev/null | tail -1)
    if [ -d "$MATLAB_ROOT/extern/engines/python" ]; then
        cd "$MATLAB_ROOT/extern/engines/python"
        pip install . 2>/dev/null || echo -e "${YELLOW}MATLAB Engine installation skipped (may require admin)${NC}"
        cd "$SCRIPT_DIR"
    fi
else
    echo -e "${YELLOW}MATLAB not found. MATLAB integration will be disabled.${NC}"
fi

# Check for Arduino CLI (optional)
echo -e "${YELLOW}Checking for Arduino CLI...${NC}"
if command -v arduino-cli &> /dev/null; then
    echo -e "${GREEN}Arduino CLI found.${NC}"
    echo -e "${YELLOW}Installing Arduino libraries...${NC}"
    arduino-cli lib install "Wire" 2>/dev/null || true
else
    echo -e "${YELLOW}Arduino CLI not found. Install it for firmware uploads.${NC}"
fi

# Check for Docker (optional, for databases)
echo -e "${YELLOW}Checking for Docker...${NC}"
if command -v docker &> /dev/null; then
    echo -e "${GREEN}Docker found.${NC}"
    
    # Ask if user wants to start databases
    echo -e "${YELLOW}Do you want to start the database containers? (y/n)${NC}"
    read -r START_DOCKER
    
    if [ "$START_DOCKER" = "y" ] || [ "$START_DOCKER" = "Y" ]; then
        echo -e "${YELLOW}Starting database containers...${NC}"
        
        # Create docker-compose file if it doesn't exist
        if [ ! -f "docker-compose.yml" ]; then
            cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  influxdb:
    image: influxdb:2.7
    ports:
      - "8086:8086"
    volumes:
      - influxdb_data:/var/lib/influxdb2
    environment:
      - DOCKER_INFLUXDB_INIT_MODE=setup
      - DOCKER_INFLUXDB_INIT_USERNAME=admin
      - DOCKER_INFLUXDB_INIT_PASSWORD=rpm_digital_twin
      - DOCKER_INFLUXDB_INIT_ORG=rpm
      - DOCKER_INFLUXDB_INIT_BUCKET=sensor_data
      - DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=rpm_influx_token

  postgres:
    image: postgres:15
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/postgres_schema.sql:/docker-entrypoint-initdb.d/init.sql
    environment:
      - POSTGRES_USER=rpm_admin
      - POSTGRES_PASSWORD=rpm_digital_twin
      - POSTGRES_DB=rpm_twin

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  influxdb_data:
  postgres_data:
  redis_data:
EOF
        fi
        
        docker-compose up -d
        echo -e "${GREEN}Database containers started.${NC}"
    fi
else
    echo -e "${YELLOW}Docker not found. Database services need manual setup.${NC}"
fi

# Run initial tests
echo -e "${YELLOW}Running initial tests...${NC}"
python -m pytest tests/ -v --tb=short 2>/dev/null || echo -e "${YELLOW}Some tests may fail without full setup.${NC}"

# Print summary
echo ""
echo -e "${GREEN}======================================"
echo "  Setup Complete!"
echo "======================================${NC}"
echo ""
echo -e "${BLUE}To activate the virtual environment:${NC}"
echo "  source venv/bin/activate"
echo ""
echo -e "${BLUE}To run the application:${NC}"
echo "  python src/main.py --mode simulation"
echo ""
echo -e "${BLUE}To run with dashboard:${NC}"
echo "  python src/main.py --mode simulation --dashboard"
echo ""
echo -e "${BLUE}To run tests:${NC}"
echo "  pytest tests/ -v"
echo ""
echo -e "${BLUE}For quick simulation test:${NC}"
echo "  python src/main.py --quick-sim --inner-rpm 2 --outer-rpm 2 --duration 60"
echo ""

# Show status
echo -e "${YELLOW}Component Status:${NC}"
echo -e "  Python Backend:    ${GREEN}✓ Ready${NC}"

if command -v matlab &> /dev/null; then
    echo -e "  MATLAB Engine:     ${GREEN}✓ Available${NC}"
else
    echo -e "  MATLAB Engine:     ${YELLOW}○ Not installed${NC}"
fi

if command -v arduino-cli &> /dev/null; then
    echo -e "  Arduino Firmware:  ${GREEN}✓ Ready to upload${NC}"
else
    echo -e "  Arduino Firmware:  ${YELLOW}○ CLI not installed${NC}"
fi

if command -v docker &> /dev/null && docker ps 2>/dev/null | grep -q influxdb; then
    echo -e "  Databases:         ${GREEN}✓ Running${NC}"
else
    echo -e "  Databases:         ${YELLOW}○ Not running${NC}"
fi

echo -e "  Unity UI:          ${YELLOW}○ Requires Unity Editor${NC}"
echo ""
