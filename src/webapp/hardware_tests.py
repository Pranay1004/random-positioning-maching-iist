"""
Hardware Tests Dashboard - FastAPI Router
==========================================
Real-time hardware test execution displaying full detailed results.
Integrates with the complete test framework from tests/hardware/.
"""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from tests.hardware.run_all import UnifiedDashboard
    from tests.hardware.test_base import TestStatus
    HAS_HARDWARE_TESTS = True
except ImportError as e:
    print(f"Warning: Could not import hardware tests: {e}")
    HAS_HARDWARE_TESTS = False

router = APIRouter(prefix="/api/hardware", tags=["hardware"])

# Global dashboard instance (cached results)
_dashboard: Optional[UnifiedDashboard] = None
_last_json_result: Optional[Dict[str, Any]] = None


async def get_dashboard() -> UnifiedDashboard:
    """Get or create the dashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = UnifiedDashboard(dry_run=False, verbose=True)
    return _dashboard


async def run_tests() -> Dict[str, Any]:
    """Run all hardware tests and return detailed results."""
    global _last_json_result
    
    if not HAS_HARDWARE_TESTS:
        return {
            "error": "Hardware tests module not available",
            "timestamp": None,
            "components": {},
            "summary": {
                "total_components": 0,
                "connected": 0,
                "disconnected": 0,
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "skipped": 0
            }
        }
    
    try:
        dashboard = await get_dashboard()
        results = await dashboard.run_all_tests()
        
        # Get the complete JSON structure from the dashboard
        json_result = dashboard.to_json()
        
        # Add skipped count
        skipped = sum(
            1 for results_list in dashboard._all_results.values()
            for r in results_list if r.status == TestStatus.SKIP
        )
        json_result["summary"]["skipped"] = skipped
        
        _last_json_result = json_result
        return json_result
        
    except Exception as e:
        print(f"Error running tests: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "timestamp": None,
            "components": {},
            "summary": {
                "total_components": 0,
                "connected": 0,
                "disconnected": 0,
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "skipped": 0
            }
        }


@router.get("/tests/status")
async def get_hardware_test_status():
    """
    Run hardware tests and return current status with all details.
    """
    return await run_tests()


@router.get("/tests/dashboard", response_class=HTMLResponse)
async def get_hardware_test_dashboard():
    """
    Serve detailed hardware test dashboard HTML.
    Shows real test results with actual packet data and communication logs.
    Nothing passes without real hardware verification.
    """
    import html as html_mod
    
    # Run tests to get latest data
    test_data = await run_tests()
    
    summary = test_data.get("summary", {})
    components = test_data.get("components", {})
    
    # Compute total packet stats
    total_tx = 0
    total_rx = 0
    total_err = 0
    for comp_data in components.values():
        for test in comp_data.get("tests", []):
            total_tx += test.get("packets_sent", 0)
            total_rx += test.get("packets_received", 0)
            total_err += test.get("packets_failed", 0)
    
    # Build summary section HTML
    summary_html = f"""
    <div class="test-summary">
        <h2>Test Summary</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="summary-number passed">{summary.get('passed', 0)}</div>
                <div class="summary-label">PASSED</div>
            </div>
            <div class="summary-item">
                <div class="summary-number failed">{summary.get('failed', 0)}</div>
                <div class="summary-label">FAILED</div>
            </div>
            <div class="summary-item">
                <div class="summary-number warned">{summary.get('warnings', 0)}</div>
                <div class="summary-label">WARNED</div>
            </div>
            <div class="summary-item">
                <div class="summary-number skipped">{summary.get('skipped', 0)}</div>
                <div class="summary-label">SKIPPED</div>
            </div>
        </div>
        <div class="summary-total">Total: {summary.get('total_tests', 0)} tests</div>
    </div>
    
    <div class="packet-summary">
        <h2>Data Transfer Summary</h2>
        <div class="packet-grid">
            <div class="packet-item tx">
                <div class="packet-number">{total_tx}</div>
                <div class="packet-label">TX PACKETS SENT</div>
            </div>
            <div class="packet-item rx">
                <div class="packet-number">{total_rx}</div>
                <div class="packet-label">RX PACKETS RECEIVED</div>
            </div>
            <div class="packet-item err">
                <div class="packet-number">{total_err}</div>
                <div class="packet-label">ERRORS</div>
            </div>
        </div>
        <div class="packet-note">
            {"No packets were transmitted — no hardware is connected." if total_tx == 0 else
             f"Success rate: {((total_tx - total_err) / total_tx * 100):.1f}%" if total_tx > 0 else ""}
        </div>
    </div>
    """
    
    # Build component cards HTML with packet details
    components_html = ""
    for comp_key, comp_data in components.items():
        tests = comp_data.get("tests", [])
        is_connected = comp_data.get("connected", False)
        status_text = "CONNECTED" if is_connected else "DISCONNECTED"
        status_color = "#00ff9d" if is_connected else "#ff6b6b"
        
        # Per-component TX/RX
        comp_tx = sum(t.get("packets_sent", 0) for t in tests)
        comp_rx = sum(t.get("packets_received", 0) for t in tests)
        comp_err = sum(t.get("packets_failed", 0) for t in tests)
        
        # Build test results table rows
        test_rows = ""
        for test in tests:
            test_status = test["status"]
            status_colors = {
                "PASS": "#00ff9d",
                "FAIL": "#ff6b6b",
                "SKIP": "#ffd60a",
                "WARN": "#ff8c00",
                "PENDING": "#888888"
            }
            badge_color = status_colors.get(test_status, "#888888")
            
            hex_color = badge_color.lstrip('#')
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            
            # Escape message for HTML
            msg = html_mod.escape(test.get('message', ''))
            
            # Packet info column
            pkt_sent = test.get('packets_sent', 0)
            pkt_recv = test.get('packets_received', 0)
            pkt_fail = test.get('packets_failed', 0)
            
            if pkt_sent > 0 or pkt_recv > 0:
                pkt_html = (f'<span class="pkt-tx">TX:{pkt_sent}</span> '
                            f'<span class="pkt-rx">RX:{pkt_recv}</span> '
                            f'<span class="pkt-err">ERR:{pkt_fail}</span>')
            else:
                pkt_html = '<span class="no-packets">—</span>'
            
            # Hex dump / data column from test.data
            data_details = ""
            test_extra_data = test.get("data", {})
            if test_extra_data:
                # Show packet hex if available
                if "packet_hex" in test_extra_data:
                    hex_str = test_extra_data["packet_hex"]
                    # Format as byte groups
                    formatted_hex = " ".join(hex_str[i:i+2] for i in range(0, min(len(hex_str), 64), 2))
                    if len(hex_str) > 64:
                        formatted_hex += "..."
                    data_details = f'<div class="hex-dump">{formatted_hex}</div>'
                else:
                    # Show other data keys as key=value pairs
                    items = []
                    for k, v in test_extra_data.items():
                        if isinstance(v, (dict, list)):
                            continue
                        items.append(f"{k}={v}")
                    if items:
                        data_details = f'<div class="test-data">{" | ".join(items[:4])}</div>'
            
            test_rows += f"""
            <tr>
                <td class="test-name">{html_mod.escape(test['name'])}</td>
                <td>
                    <span class="test-badge" style="background: rgba({r},{g},{b},0.2); color: {badge_color}; border: 1px solid {badge_color};">
                        {test_status}
                    </span>
                </td>
                <td class="test-message">{msg}{data_details}</td>
                <td class="test-packets">{pkt_html}</td>
                <td class="test-duration">{test.get('duration_ms', 0):.1f}ms</td>
            </tr>
            """
        
        hex_sc = status_color.lstrip('#')
        sr, sg, sb = int(hex_sc[0:2], 16), int(hex_sc[2:4], 16), int(hex_sc[4:6], 16)
        
        components_html += f"""
        <div class="component-card">
            <div class="component-header">
                <div class="component-title">
                    <h3>{html_mod.escape(comp_data['name'])}</h3>
                    <span class="component-type">{comp_data['connection']}</span>
                </div>
                <div class="status-badge" style="background: rgba({sr},{sg},{sb},0.2); color: {status_color}; border: 1px solid {status_color};">
                    {status_text}
                </div>
            </div>
            <div class="component-info">
                <div>Port: <strong>{html_mod.escape(str(comp_data.get('port', 'N/A')))}</strong></div>
                <div>TX: <strong>{comp_tx}</strong> | RX: <strong>{comp_rx}</strong> | ERR: <strong>{comp_err}</strong></div>
                {f"<div>Latency: <strong>{comp_data.get('latency_ms')} ms</strong></div>" if comp_data.get('latency_ms') else ""}
            </div>
            <table class="test-results-table">
                <thead>
                    <tr>
                        <th>TEST NAME</th>
                        <th>STATUS</th>
                        <th>MESSAGE / DATA</th>
                        <th>PACKETS</th>
                        <th>DURATION</th>
                    </tr>
                </thead>
                <tbody>
                    {test_rows}
                </tbody>
            </table>
        </div>
        """
    
    css = """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Roboto Mono', monospace;
            background: linear-gradient(135deg, #050508 0%, #0a0a14 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .header {
            margin-bottom: 30px;
        }
        
        .back-btn {
            display: inline-block;
            padding: 10px 20px;
            background: rgba(0, 229, 255, 0.1);
            border: 2px solid #00e5ff;
            color: #00e5ff;
            text-decoration: none;
            border-radius: 4px;
            font-weight: 700;
            letter-spacing: 1px;
            margin-bottom: 20px;
            transition: all 0.3s;
            font-size: 0.85em;
        }
        
        .back-btn:hover {
            background: #00e5ff;
            color: #050508;
        }
        
        h1 {
            font-family: 'Orbitron', sans-serif;
            color: #00e5ff;
            font-size: 2.2em;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 5px;
        }
        
        .subtitle {
            color: #888;
            font-size: 0.9em;
        }
        
        .test-summary, .packet-summary {
            background: rgba(255, 255, 255, 0.03);
            border: 2px solid rgba(0, 229, 255, 0.4);
            border-radius: 8px;
            padding: 25px;
            margin-bottom: 25px;
        }
        
        .test-summary h2, .packet-summary h2 {
            font-family: 'Orbitron', sans-serif;
            color: #00e5ff;
            font-size: 1.4em;
            margin-bottom: 15px;
            text-align: center;
            letter-spacing: 2px;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 15px;
        }
        
        .summary-item {
            text-align: center;
            padding: 12px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 4px;
        }
        
        .summary-number {
            font-size: 2.5em;
            font-weight: 700;
            font-family: 'Orbitron', sans-serif;
            margin-bottom: 5px;
        }
        
        .summary-number.passed { color: #00ff9d; }
        .summary-number.failed { color: #ff6b6b; }
        .summary-number.warned { color: #ff8c00; }
        .summary-number.skipped { color: #888; }
        
        .summary-label {
            color: #888;
            font-size: 0.8em;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 700;
        }
        
        .summary-total {
            text-align: center;
            color: #888;
            font-size: 0.85em;
            padding-top: 12px;
            border-top: 1px solid rgba(0, 229, 255, 0.2);
        }
        
        .packet-summary {
            border-color: rgba(255, 107, 107, 0.4);
        }
        
        .packet-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 10px;
        }
        
        .packet-item {
            text-align: center;
            padding: 15px;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.05);
        }
        
        .packet-number {
            font-size: 2.2em;
            font-weight: 700;
            font-family: 'Orbitron', sans-serif;
        }
        
        .packet-item.tx .packet-number { color: #00e5ff; }
        .packet-item.rx .packet-number { color: #00ff9d; }
        .packet-item.err .packet-number { color: #ff6b6b; }
        
        .packet-label {
            color: #888;
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
            margin-top: 4px;
        }
        
        .packet-note {
            text-align: center;
            color: #ff6b6b;
            font-size: 0.85em;
            padding-top: 10px;
            border-top: 1px solid rgba(255, 107, 107, 0.2);
            font-weight: 600;
        }
        
        .components-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }
        
        .component-card {
            background: rgba(255, 255, 255, 0.03);
            border: 2px solid rgba(0, 229, 255, 0.3);
            border-radius: 8px;
            padding: 18px;
            transition: all 0.3s;
        }
        
        .component-card:hover {
            border-color: #00e5ff;
            box-shadow: 0 0 20px rgba(0, 229, 255, 0.15);
        }
        
        .component-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(0, 229, 255, 0.2);
        }
        
        .component-title h3 {
            color: #00e5ff;
            font-size: 1.05em;
            margin-bottom: 3px;
            font-family: 'Orbitron', sans-serif;
        }
        
        .component-type {
            color: #666;
            font-size: 0.8em;
        }
        
        .status-badge {
            padding: 5px 12px;
            border-radius: 4px;
            font-size: 0.72em;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .component-info {
            color: #aaa;
            font-size: 0.82em;
            margin-bottom: 12px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .test-results-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 8px;
        }
        
        .test-results-table thead {
            background: rgba(0, 229, 255, 0.1);
            border-bottom: 2px solid rgba(0, 229, 255, 0.3);
        }
        
        .test-results-table th {
            padding: 8px 10px;
            text-align: left;
            color: #00e5ff;
            font-weight: 700;
            font-size: 0.72em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .test-results-table td {
            padding: 7px 10px;
            border-bottom: 1px solid rgba(0, 229, 255, 0.08);
            font-size: 0.82em;
            vertical-align: top;
        }
        
        .test-name {
            white-space: nowrap;
        }
        
        .test-badge {
            padding: 2px 7px;
            border-radius: 3px;
            font-size: 0.72em;
            font-weight: 700;
            white-space: nowrap;
        }
        
        .test-message {
            color: #aaa;
            max-width: 500px;
        }
        
        .test-packets {
            white-space: nowrap;
            font-size: 0.78em;
        }
        
        .pkt-tx { color: #00e5ff; margin-right: 6px; }
        .pkt-rx { color: #00ff9d; margin-right: 6px; }
        .pkt-err { color: #ff6b6b; }
        .no-packets { color: #444; }
        
        .hex-dump {
            font-family: 'Roboto Mono', monospace;
            font-size: 0.78em;
            color: #00e5ff;
            background: rgba(0, 229, 255, 0.05);
            border: 1px solid rgba(0, 229, 255, 0.15);
            border-radius: 3px;
            padding: 4px 8px;
            margin-top: 4px;
            word-break: break-all;
            line-height: 1.4;
        }
        
        .test-data {
            font-size: 0.8em;
            color: #666;
            margin-top: 3px;
        }
        
        .test-duration {
            color: #555;
            text-align: right;
            white-space: nowrap;
        }
        
        .refresh-btn {
            display: block;
            margin: 30px auto;
            padding: 12px 40px;
            background: rgba(0, 229, 255, 0.1);
            border: 2px solid #00e5ff;
            color: #00e5ff;
            text-decoration: none;
            border-radius: 4px;
            font-weight: 700;
            letter-spacing: 1px;
            cursor: pointer;
            font-size: 0.85em;
            transition: all 0.3s;
            font-family: 'Roboto Mono', monospace;
            text-transform: uppercase;
        }
        
        .refresh-btn:hover {
            background: #00e5ff;
            color: #050508;
        }
        
        .footer {
            text-align: center;
            color: #555;
            font-size: 0.75em;
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid rgba(0, 229, 255, 0.15);
        }
        
        .hw-notice {
            background: rgba(255, 107, 107, 0.1);
            border: 1px solid rgba(255, 107, 107, 0.3);
            border-radius: 6px;
            padding: 15px 20px;
            margin-bottom: 25px;
            color: #ff6b6b;
            font-size: 0.85em;
            line-height: 1.5;
        }
        
        .hw-notice strong {
            color: #ff8c8c;
        }
    """

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hardware Tests Dashboard - RPM Digital Twin</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto+Mono:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>{css}</style>
</head>
<body>
    <div class="container">
        <div class="header">
            <a href="/" class="back-btn">&#8592; BACK TO DASHBOARD</a>
            <h1>HARDWARE TESTS DASHBOARD</h1>
            <p class="subtitle">Real hardware verification &mdash; no simulated results</p>
        </div>
        
        {"<div class='hw-notice'><strong>NO HARDWARE DETECTED:</strong> All tests that require physical hardware connections (GPIO, Serial, I2C) are reporting SKIP or FAIL. Connect the actual RPM hardware (RPi Pico W, NEMA motors, slip rings) and re-run to see PASS results with real packet data.</div>" if summary.get('passed', 0) <= 3 and total_tx == 0 else ""}
        
        {summary_html}
        
        <div class="components-grid">
            {components_html}
        </div>
        
        <button class="refresh-btn" onclick="location.reload()">REFRESH TESTS</button>
        
        <div class="footer">
            Last run: {test_data.get('timestamp', 'Never')} | 
            Only tests with verified hardware communication report PASS
        </div>
    </div>
</body>
</html>
    """
    return html


