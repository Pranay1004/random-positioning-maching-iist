#!/usr/bin/env bash
# =============================================================================
# RPM Digital Twin - Raspberry Pi 5 Deployment Script
# =============================================================================
# Sets up the RPi 5 as the RPM brain: FastAPI server, auto-start service,
# network configuration for local web dashboard access.
#
# Usage:
#   chmod +x deploy/rpi5_setup.sh
#   sudo ./deploy/rpi5_setup.sh
#
# What it does:
#   1. System update & dependencies
#   2. Python 3.11+ virtual environment
#   3. Install project requirements
#   4. Create systemd service for auto-start
#   5. Configure network for 0.0.0.0:8080 access
#   6. Enable I2C, SPI, UART interfaces
#   7. GPIO permissions setup
# =============================================================================

set -euo pipefail

# Colors
RED='\033[91m'
GREEN='\033[92m'
YELLOW='\033[93m'
CYAN='\033[96m'
BOLD='\033[1m'
RESET='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_NAME="rpm-digital-twin"
SERVICE_USER="${SUDO_USER:-pi}"
PYTHON_MIN="3.11"
WEB_PORT=8080

echo -e "${CYAN}${BOLD}========================================${RESET}"
echo -e "${CYAN}  RPM Digital Twin - RPi 5 Deployment${RESET}"
echo -e "${CYAN}${BOLD}========================================${RESET}"
echo -e "  Project: ${PROJECT_DIR}"
echo -e "  User:    ${SERVICE_USER}"
echo -e "  Port:    ${WEB_PORT}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}ERROR: Run with sudo${RESET}"
    exit 1
fi

# Check if on RPi
if [ ! -f /proc/device-tree/model ]; then
    echo -e "${YELLOW}WARNING: Not running on Raspberry Pi${RESET}"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# =============================================================================
# 1. System Update & Dependencies
# =============================================================================
echo -e "\n${CYAN}[1/7] System Update & Dependencies${RESET}"

apt-get update -qq
apt-get install -y -qq \
    python3 python3-pip python3-venv \
    i2c-tools \
    libatlas-base-dev \
    git

echo -e "${GREEN}  System packages installed${RESET}"

# =============================================================================
# 2. Python Virtual Environment
# =============================================================================
echo -e "\n${CYAN}[2/7] Python Environment${RESET}"

VENV_DIR="${PROJECT_DIR}/.venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}  Virtual environment created: ${VENV_DIR}${RESET}"
else
    echo -e "${GREEN}  Virtual environment exists: ${VENV_DIR}${RESET}"
fi

# Verify Python version
PY_VERSION=$("${VENV_DIR}/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo -e "  Python version: ${PY_VERSION}"

# =============================================================================
# 3. Install Requirements
# =============================================================================
echo -e "\n${CYAN}[3/7] Installing Python Packages${RESET}"

"${VENV_DIR}/bin/pip" install --upgrade pip -q
"${VENV_DIR}/bin/pip" install -r "${PROJECT_DIR}/requirements.txt" -q

echo -e "${GREEN}  All packages installed${RESET}"

# =============================================================================
# 4. Systemd Service
# =============================================================================
echo -e "\n${CYAN}[4/7] Creating Systemd Service${RESET}"

cat > /etc/systemd/system/${SERVICE_NAME}.service << EOF
[Unit]
Description=RPM Digital Twin - Web Dashboard & Control
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
Group=${SERVICE_USER}
WorkingDirectory=${PROJECT_DIR}
ExecStart=${VENV_DIR}/bin/python src/main.py --mode simulation --dashboard
ExecStop=/bin/kill -SIGINT \$MAINPID
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=PATH=${VENV_DIR}/bin:/usr/local/bin:/usr/bin
Environment=PYTHONPATH=${PROJECT_DIR}/src

# Security hardening
NoNewPrivileges=false
ProtectSystem=false
ProtectHome=false

# GPIO access requires these not to be restricted
ProtectKernelModules=false
DeviceAllow=/dev/ttyUSB* rw
DeviceAllow=/dev/ttyACM* rw
DeviceAllow=/dev/gpiochip* rw
DeviceAllow=/dev/i2c-* rw
DeviceAllow=/dev/spidev* rw

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${SERVICE_NAME}

echo -e "${GREEN}  Service created: ${SERVICE_NAME}${RESET}"
echo -e "  ${YELLOW}Start: sudo systemctl start ${SERVICE_NAME}${RESET}"
echo -e "  ${YELLOW}Logs:  journalctl -u ${SERVICE_NAME} -f${RESET}"

# =============================================================================
# 5. Network / Firewall
# =============================================================================
echo -e "\n${CYAN}[5/7] Network Configuration${RESET}"

# Allow port 8080 through firewall (if ufw is active)
if command -v ufw &> /dev/null; then
    ufw allow ${WEB_PORT}/tcp 2>/dev/null || true
    echo -e "${GREEN}  Firewall: Port ${WEB_PORT} allowed${RESET}"
else
    echo -e "  No firewall detected (ufw not installed)"
fi

# Get IP addresses
IP_WLAN=$(ip -4 addr show wlan0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || echo "N/A")
IP_ETH=$(ip -4 addr show eth0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || echo "N/A")

echo -e "  Dashboard accessible at:"
echo -e "    ${GREEN}http://localhost:${WEB_PORT}${RESET}"
if [ "$IP_WLAN" != "N/A" ]; then
    echo -e "    ${GREEN}http://${IP_WLAN}:${WEB_PORT}${RESET}  (WiFi)"
fi
if [ "$IP_ETH" != "N/A" ]; then
    echo -e "    ${GREEN}http://${IP_ETH}:${WEB_PORT}${RESET}  (Ethernet)"
fi

# =============================================================================
# 6. Enable Interfaces (I2C, SPI, UART)
# =============================================================================
echo -e "\n${CYAN}[6/7] Hardware Interfaces${RESET}"

# Enable I2C
if ! grep -q "^dtparam=i2c_arm=on" /boot/firmware/config.txt 2>/dev/null; then
    echo "dtparam=i2c_arm=on" >> /boot/firmware/config.txt
    echo -e "${GREEN}  I2C enabled${RESET}"
else
    echo -e "  I2C already enabled"
fi

# Enable SPI
if ! grep -q "^dtparam=spi=on" /boot/firmware/config.txt 2>/dev/null; then
    echo "dtparam=spi=on" >> /boot/firmware/config.txt
    echo -e "${GREEN}  SPI enabled${RESET}"
else
    echo -e "  SPI already enabled"
fi

# Enable UART
if ! grep -q "^enable_uart=1" /boot/firmware/config.txt 2>/dev/null; then
    echo "enable_uart=1" >> /boot/firmware/config.txt
    echo -e "${GREEN}  UART enabled${RESET}"
else
    echo -e "  UART already enabled"
fi

# =============================================================================
# 7. GPIO Permissions
# =============================================================================
echo -e "\n${CYAN}[7/7] GPIO Permissions${RESET}"

# Add user to required groups
usermod -aG gpio,i2c,spi,dialout "${SERVICE_USER}" 2>/dev/null || true

# GPIO udev rules for non-root access
cat > /etc/udev/rules.d/99-gpio.rules << EOF
SUBSYSTEM=="gpio", KERNEL=="gpiochip*", MODE="0660", GROUP="gpio"
EOF

udevadm control --reload-rules 2>/dev/null || true

echo -e "${GREEN}  User '${SERVICE_USER}' added to gpio, i2c, spi, dialout groups${RESET}"

# =============================================================================
# Summary
# =============================================================================
echo -e "\n${CYAN}${BOLD}========================================${RESET}"
echo -e "${GREEN}${BOLD}  DEPLOYMENT COMPLETE${RESET}"
echo -e "${CYAN}${BOLD}========================================${RESET}"
echo ""
echo -e "  ${BOLD}Quick Commands:${RESET}"
echo -e "    Start server:   sudo systemctl start ${SERVICE_NAME}"
echo -e "    Stop server:    sudo systemctl stop ${SERVICE_NAME}"
echo -e "    View logs:      journalctl -u ${SERVICE_NAME} -f"
echo -e "    Server status:  systemctl status ${SERVICE_NAME}"
echo -e "    Run manually:   cd ${PROJECT_DIR} && python src/main.py"
echo -e ""
echo -e "  ${BOLD}Hardware Tests:${RESET}"
echo -e "    All tests:      python -m tests.hardware.run_all"
echo -e "    RPi self-test:  python -m tests.hardware.test_rpi5"
echo -e "    Pico W test:    python -m tests.hardware.test_pico_w"
echo -e "    Motor test:     python -m tests.hardware.test_motors --dry-run"
echo -e "    Slip ring test: python -m tests.hardware.test_slip_rings"
echo -e ""
echo -e "  ${YELLOW}NOTE: Reboot recommended to apply I2C/SPI/UART changes.${RESET}"
echo ""
