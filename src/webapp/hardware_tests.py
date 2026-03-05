"""
Hardware Test Results Web Dashboard
====================================
Real-time display of all 7 hardware component tests with live status updates.
Runs ACTUAL tests — no hardcoded or fake data.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root so test imports work
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

router = APIRouter(prefix="/api/hardware", tags=["hardware"])


@router.get("/tests/status")
async def get_hardware_tests() -> Dict[str, Any]:
    """
    Run ACTUAL hardware tests and return real results.
    No hardcoded or fake data — every result comes from a real test execution.
    """
    from tests.hardware.test_base import TestStatus
    from tests.hardware.test_pico_w import PicoWTest
    from tests.hardware.test_rpi5 import RPi5DiagnosticsTest
    from tests.hardware.test_motors import MotorTest
    from tests.hardware.test_slip_rings import SlipRingTest

    from datetime import datetime

    timestamp = datetime.now().isoformat()
    components = []

    # 1. RPi Pico W
    pico = PicoWTest(port=None, verbose=False)
    pico_results = await pico.run_tests()
    components.append(_summarize("RPi Pico W", pico_results))

    # 2. RPi 5
    rpi5 = RPi5DiagnosticsTest(verbose=False)
    rpi5_results = await rpi5.run_tests()
    components.append(_summarize("RPi 5 (8GB)", rpi5_results))

    # 3-5. Motors (no dry-run — report truth)
    for motor_key, label in [("inner", "NEMA 23 — Inner Frame"),
                             ("outer", "NEMA 23 — Outer Frame"),
                             ("nema24", "NEMA 24 — Third Axis")]:
        motor = MotorTest(motor_key=motor_key, serial_port=None,
                         dry_run=False, verbose=False)
        motor_results = await motor.run_tests()
        components.append(_summarize(label, motor_results))

    # 6-7. Slip rings
    for ring_id, label in [("inner", "Slip Ring — Inner"),
                           ("outer", "Slip Ring — Outer")]:
        slip = SlipRingTest(ring_id=ring_id, serial_port=None, verbose=False)
        slip_results = await slip.run_tests()
        components.append(_summarize(label, slip_results))

    # Totals
    total_p = sum(c["passed"] for c in components)
    total_f = sum(c["failed"] for c in components)
    total_w = sum(c["warned"] for c in components)
    total_s = sum(c["skipped"] for c in components)

    return {
        "timestamp": timestamp,
        "total_tests": total_p + total_f + total_w + total_s,
        "passed": total_p,
        "failed": total_f,
        "warned": total_w,
        "skipped": total_s,
        "components": components,
    }


def _summarize(name: str, results) -> Dict[str, Any]:
    """Summarize test results for a component — counts only real statuses."""
    from tests.hardware.test_base import TestStatus

    passed = sum(1 for r in results if r.status == TestStatus.PASS)
    failed = sum(1 for r in results if r.status == TestStatus.FAIL)
    warned = sum(1 for r in results if r.status == TestStatus.WARN)
    skipped = sum(1 for r in results if r.status == TestStatus.SKIP)

    # Determine connection status honestly
    if failed > 0:
        status = "DISCONNECTED"
    elif passed > 0:
        # Check if any PASS was a real hardware test (not just software/config)
        real_hw = sum(
            1 for r in results
            if r.status == TestStatus.PASS and
            not any(kw in r.name.lower() for kw in ["config", "calculat", "protocol", "software", "package", "environment", "readiness"])
        )
        status = "CONNECTED" if real_hw > 0 else "NOT VERIFIED"
    else:
        status = "NOT TESTED"

    return {
        "name": name,
        "status": status,
        "passed": passed,
        "failed": failed,
        "warned": warned,
        "skipped": skipped,
        "tests": [
            {
                "name": r.name,
                "status": r.status.value,
                "message": r.message,
            }
            for r in results
        ],
    }


@router.get("/tests/dashboard")
async def get_hardware_dashboard() -> HTMLResponse:
    """
    Return HTML dashboard for hardware test results.
    Includes live status display, color coding, and detail panels.
    """
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>RPM Hardware Tests Dashboard</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            
            body {
                font-family: 'Roboto Mono', monospace;
                background: linear-gradient(135deg, #050508 0%, #0a0a12 100%);
                color: #00e5ff;
                padding: 20px;
                min-height: 100vh;
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
                border-bottom: 2px solid #00ff9d;
                padding-bottom: 20px;
            }
            
            .header h1 {
                font-size: 2.5em;
                color: #00ff9d;
                margin-bottom: 10px;
                text-shadow: 0 0 20px rgba(0, 255, 157, 0.5);
            }
            
            .header p {
                color: #888;
                font-size: 0.9em;
            }
            
            .controls {
                display: flex;
                gap: 10px;
                margin-bottom: 30px;
                flex-wrap: wrap;
            }
            
            button {
                padding: 10px 20px;
                background: #00ff9d;
                border: none;
                color: #050508;
                cursor: pointer;
                font-weight: bold;
                border-radius: 3px;
                font-family: 'Roboto Mono', monospace;
                transition: all 0.3s;
            }
            
            button:hover {
                transform: scale(1.05);
                box-shadow: 0 0 20px rgba(0, 255, 157, 0.5);
            }
            
            .status-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }
            
            .component-card {
                background: rgba(0, 0, 0, 0.5);
                border: 2px solid #00e5ff;
                border-radius: 5px;
                padding: 20px;
                transition: all 0.3s;
            }
            
            .component-card:hover {
                border-color: #00ff9d;
                box-shadow: 0 0 15px rgba(0, 255, 157, 0.3);
            }
            
            .component-name {
                font-size: 1.2em;
                font-weight: bold;
                color: #00ff9d;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .status-indicator {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                display: inline-block;
            }
            
            .status-connected {
                background: #00ff9d;
                box-shadow: 0 0 10px #00ff9d;
            }
            
            .status-disconnected {
                background: #ff4444;
                box-shadow: 0 0 10px #ff4444;
            }
            
            .status-not-verified {
                background: #666;
                box-shadow: 0 0 10px #666;
            }
            
            .test-results {
                display: flex;
                gap: 15px;
                margin: 15px 0;
                font-size: 0.85em;
            }
            
            .test-stat {
                padding: 8px 12px;
                background: rgba(0, 0, 0, 0.7);
                border-radius: 3px;
                text-align: center;
            }
            
            .test-stat.pass { color: #00ff9d; }
            .test-stat.fail { color: #ff4444; }
            .test-stat.warn { color: #ffaa00; }
            .test-stat.skip { color: #666; }
            
            .overall-summary {
                background: rgba(0, 0, 0, 0.5);
                border: 2px solid #00ff9d;
                border-radius: 5px;
                padding: 30px;
                text-align: center;
            }
            
            .summary-stat {
                display: inline-block;
                margin: 0 30px;
                font-size: 1.5em;
                font-weight: bold;
            }
            
            .summary-stat .label {
                font-size: 0.7em;
                color: #888;
                display: block;
            }
            
            .loading {
                text-align: center;
                padding: 40px;
                color: #00ff9d;
                font-size: 1.2em;
            }
            
            .spinner {
                border: 3px solid #00ff9d;
                border-top: 3px solid transparent;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 0.8s linear infinite;
                margin: 0 auto 20px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .timestamp {
                color: #666;
                font-size: 0.85em;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 RPM HARDWARE TESTS</h1>
                <p>Real-time Hardware Connection Diagnostics</p>
            </div>
            
            <div class="controls">
                <button onclick="runTests()">▶ Run Tests</button>
                <button onclick="autoRefresh()">🔄 Auto Refresh (10s)</button>
                <button onclick="exportJSON()">📥 Export JSON</button>
                <span style="color: #888; padding: 10px;">
                    Last updated: <span id="lastUpdate">--:--:--</span>
                </span>
            </div>
            
            <div id="content">
                <div class="loading">
                    <div class="spinner"></div>
                    Loading hardware status...
                </div>
            </div>
            
            <div class="timestamp">
                Auto-refresh: <span id="autoRefreshStatus">off</span> 
                | Page loaded: <span id="pageLoad">--</span>
            </div>
        </div>
        
        <script>
            let autoRefreshInterval = null;
            let testData = null;
            
            function formatTime(dt) {
                if (!dt) return '--:--:--';
                return new Date(dt).toLocaleTimeString('en-GB');
            }
            
            async function runTests() {
                const content = document.getElementById('content');
                content.innerHTML = '<div class="loading"><div class="spinner"></div>Running tests...</div>';
                
                try {
                    const response = await fetch('/api/hardware/tests/status');
                    testData = await response.json();
                    renderResults(testData);
                    document.getElementById('lastUpdate').textContent = formatTime(new Date());
                    document.getElementById('pageLoad').textContent = formatTime(new Date());
                } catch (e) {
                    content.innerHTML = '<div class="loading" style="color: #ff4444;">❌ Failed to fetch test results: ' + e.message + '</div>';
                }
            }
            
            function renderResults(data) {
                if (!data) return;
                
                const content = document.getElementById('content');
                
                if (data.components && data.components.length > 0) {
                    // Component grid
                    let html = '<div class="status-grid">';
                    
                    for (const comp of data.components) {
                        const statusClass = comp.status === 'CONNECTED' ? 'status-connected' : 
                                          comp.status === 'DISCONNECTED' ? 'status-disconnected' :
                                          comp.status === 'NOT VERIFIED' ? 'status-not-verified' : 'status-warning';
                        
                        html += `
                            <div class="component-card">
                                <div class="component-name">
                                    <span class="status-indicator ${statusClass}"></span>
                                    ${comp.name}
                                </div>
                                <div style="color: ${comp.status === 'CONNECTED' ? '#00ff9d' : comp.status === 'DISCONNECTED' ? '#ff4444' : '#888'}; font-size: 0.85em; margin-bottom: 10px; font-weight: bold;">${comp.status}</div>
                                <div class="test-results">
                                    <div class="test-stat pass">&check; ${comp.passed || 0}</div>
                                    <div class="test-stat fail">&cross; ${comp.failed || 0}</div>
                                    <div class="test-stat warn">&excl; ${comp.warned || 0}</div>
                                    <div class="test-stat skip">&oslash; ${comp.skipped || 0}</div>
                                </div>
                            </div>
                        `;
                    }
                    html += '</div>';
                    
                    // Detailed test results per component
                    for (const comp of data.components) {
                        if (comp.tests && comp.tests.length > 0) {
                            html += '<div style="margin-bottom: 20px;">';
                            html += '<h3 style="color: #00e5ff; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 5px;">' + comp.name + '</h3>';
                            for (const test of comp.tests) {
                                const color = test.status === 'PASS' ? '#00ff9d' : 
                                             test.status === 'FAIL' ? '#ff4444' : 
                                             test.status === 'WARN' ? '#ffaa00' : '#666';
                                const icon = test.status === 'PASS' ? '&check;' : 
                                            test.status === 'FAIL' ? '&cross;' : 
                                            test.status === 'WARN' ? '!' : '&oslash;';
                                html += '<div style="padding: 4px 10px; font-size: 0.85em;">';
                                html += '<span style="color: ' + color + '; font-weight: bold; width: 60px; display: inline-block;">[' + test.status + ']</span> ';
                                html += '<span style="color: #ccc;">' + test.name + '</span>';
                                if (test.message) {
                                    html += '<div style="color: #888; padding-left: 70px; font-size: 0.9em;">' + test.message + '</div>';
                                }
                                html += '</div>';
                            }
                            html += '</div>';
                        }
                    }
                    
                    // Overall summary
                    html += `
                        <div class="overall-summary">
                            <h2 style="margin-bottom: 20px; color: #00ff9d;">Test Summary</h2>
                            <div class="summary-stat" style="color: #00ff9d;">
                                ${data.passed || 0}
                                <span class="label">PASSED</span>
                            </div>
                            <div class="summary-stat" style="color: #ff4444;">
                                ${data.failed || 0}
                                <span class="label">FAILED</span>
                            </div>
                            <div class="summary-stat" style="color: #ffaa00;">
                                ${data.warned || 0}
                                <span class="label">WARNED</span>
                            </div>
                            <div class="summary-stat" style="color: #666;">
                                ${data.skipped || 0}
                                <span class="label">SKIPPED</span>
                            </div>
                            <div style="margin-top: 20px; color: #888;">
                                Total: ${data.total_tests || 0} tests
                            </div>
                        </div>
                    `;
                    
                    content.innerHTML = html;
                } else {
                    content.innerHTML = '<div class="loading">No test data available</div>';
                }
            }
            
            function autoRefresh() {
                if (autoRefreshInterval) {
                    clearInterval(autoRefreshInterval);
                    autoRefreshInterval = null;
                    document.getElementById('autoRefreshStatus').textContent = 'off';
                } else {
                    document.getElementById('autoRefreshStatus').textContent = '10s';
                    runTests();
                    autoRefreshInterval = setInterval(runTests, 10000);
                }
            }
            
            function exportJSON() {
                if (!testData) {
                    alert('Run tests first');
                    return;
                }
                const json = JSON.stringify(testData, null, 2);
                const blob = new Blob([json], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `hardware_tests_${new Date().toISOString().slice(0, 10)}.json`;
                a.click();
            }
            
            // Load on page load
            document.addEventListener('DOMContentLoaded', () => {
                runTests();
            });
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)
