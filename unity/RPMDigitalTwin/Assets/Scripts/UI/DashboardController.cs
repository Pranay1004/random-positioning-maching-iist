/*
 * RPM Digital Twin - Dashboard Controller
 * ========================================
 * 
 * Main dashboard UI controller with Tesla/SpaceX-style interface.
 * 
 * Features:
 * - Real-time gauge animations
 * - Gravity vector visualization
 * - Motor control interface
 * - Emergency stop functionality
 */

using System;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace RPMDigitalTwin.UI
{
    /// <summary>
    /// Main dashboard controller for the RPM Digital Twin UI
    /// </summary>
    public class DashboardController : MonoBehaviour
    {
        #region Inspector Fields

        [Header("Connection")]
        [SerializeField] private BackendConnectionManager connectionManager;

        [Header("Gauge References")]
        [SerializeField] private CircularGauge innerFrameGauge;
        [SerializeField] private CircularGauge outerFrameGauge;
        [SerializeField] private CircularGauge instantaneousGGauge;
        [SerializeField] private CircularGauge timeAveragedGGauge;

        [Header("Text Displays")]
        [SerializeField] private TextMeshProUGUI innerRPMText;
        [SerializeField] private TextMeshProUGUI outerRPMText;
        [SerializeField] private TextMeshProUGUI instantaneousGText;
        [SerializeField] private TextMeshProUGUI timeAveragedGText;
        [SerializeField] private TextMeshProUGUI connectionStatusText;
        [SerializeField] private TextMeshProUGUI timestampText;

        [Header("Control Sliders")]
        [SerializeField] private Slider innerRPMSlider;
        [SerializeField] private Slider outerRPMSlider;

        [Header("Buttons")]
        [SerializeField] private Button startButton;
        [SerializeField] private Button stopButton;
        [SerializeField] private Button emergencyStopButton;

        [Header("3D Visualization")]
        [SerializeField] private Transform innerFrameModel;
        [SerializeField] private Transform outerFrameModel;
        [SerializeField] private LineRenderer gravityVectorLine;

        [Header("Status Indicators")]
        [SerializeField] private Image connectionIndicator;
        [SerializeField] private Image innerMotorIndicator;
        [SerializeField] private Image outerMotorIndicator;

        [Header("Colors")]
        [SerializeField] private Color connectedColor = new Color(0f, 0.8f, 0.3f);
        [SerializeField] private Color disconnectedColor = new Color(0.8f, 0.2f, 0.2f);
        [SerializeField] private Color runningColor = new Color(0f, 0.6f, 1f);
        [SerializeField] private Color idleColor = new Color(0.4f, 0.4f, 0.4f);

        #endregion

        #region Private Fields

        private RPMStateData _currentState;
        private float _targetInnerRPM;
        private float _targetOuterRPM;
        private bool _motorsRunning;

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            // Find connection manager if not assigned
            if (connectionManager == null)
            {
                connectionManager = FindObjectOfType<BackendConnectionManager>();
            }
        }

        private void Start()
        {
            SetupUI();
            RegisterEventListeners();
        }

        private void Update()
        {
            UpdateTimestamp();
        }

        private void OnDestroy()
        {
            UnregisterEventListeners();
        }

        #endregion

        #region Setup

        private void SetupUI()
        {
            // Setup gauges
            if (innerFrameGauge != null)
            {
                innerFrameGauge.SetRange(-10f, 10f);
                innerFrameGauge.SetValue(0f);
            }

            if (outerFrameGauge != null)
            {
                outerFrameGauge.SetRange(-10f, 10f);
                outerFrameGauge.SetValue(0f);
            }

            if (instantaneousGGauge != null)
            {
                instantaneousGGauge.SetRange(0f, 1f);
                instantaneousGGauge.SetValue(0f);
            }

            if (timeAveragedGGauge != null)
            {
                timeAveragedGGauge.SetRange(0f, 0.1f);
                timeAveragedGGauge.SetValue(0f);
            }

            // Setup sliders
            if (innerRPMSlider != null)
            {
                innerRPMSlider.minValue = -10f;
                innerRPMSlider.maxValue = 10f;
                innerRPMSlider.value = 0f;
                innerRPMSlider.onValueChanged.AddListener(OnInnerRPMSliderChanged);
            }

            if (outerRPMSlider != null)
            {
                outerRPMSlider.minValue = -10f;
                outerRPMSlider.maxValue = 10f;
                outerRPMSlider.value = 0f;
                outerRPMSlider.onValueChanged.AddListener(OnOuterRPMSliderChanged);
            }

            // Setup buttons
            if (startButton != null)
                startButton.onClick.AddListener(OnStartButtonClicked);

            if (stopButton != null)
                stopButton.onClick.AddListener(OnStopButtonClicked);

            if (emergencyStopButton != null)
                emergencyStopButton.onClick.AddListener(OnEmergencyStopClicked);

            // Initial state
            UpdateConnectionStatus(ConnectionStatus.Disconnected);
        }

        private void RegisterEventListeners()
        {
            if (connectionManager != null)
            {
                connectionManager.OnStateUpdate.AddListener(OnStateUpdated);
                connectionManager.OnConnectionStatusChanged.AddListener(UpdateConnectionStatus);
            }
        }

        private void UnregisterEventListeners()
        {
            if (connectionManager != null)
            {
                connectionManager.OnStateUpdate.RemoveListener(OnStateUpdated);
                connectionManager.OnConnectionStatusChanged.RemoveListener(UpdateConnectionStatus);
            }
        }

        #endregion

        #region Event Handlers

        private void OnStateUpdated(RPMStateData state)
        {
            _currentState = state;
            UpdateDashboard();
        }

        private void OnInnerRPMSliderChanged(float value)
        {
            _targetInnerRPM = value;
            
            if (_motorsRunning)
            {
                connectionManager?.SetInnerFrameRPM(value);
            }
        }

        private void OnOuterRPMSliderChanged(float value)
        {
            _targetOuterRPM = value;
            
            if (_motorsRunning)
            {
                connectionManager?.SetOuterFrameRPM(value);
            }
        }

        private void OnStartButtonClicked()
        {
            _motorsRunning = true;
            connectionManager?.SetInnerFrameRPM(_targetInnerRPM);
            connectionManager?.SetOuterFrameRPM(_targetOuterRPM);
            
            UpdateMotorIndicators(true, true);
            Debug.Log("[Dashboard] Motors started");
        }

        private void OnStopButtonClicked()
        {
            _motorsRunning = false;
            connectionManager?.SetInnerFrameRPM(0f);
            connectionManager?.SetOuterFrameRPM(0f);
            
            UpdateMotorIndicators(false, false);
            Debug.Log("[Dashboard] Motors stopped");
        }

        private void OnEmergencyStopClicked()
        {
            _motorsRunning = false;
            connectionManager?.EmergencyStop();
            
            innerRPMSlider.value = 0f;
            outerRPMSlider.value = 0f;
            
            UpdateMotorIndicators(false, false);
            Debug.LogWarning("[Dashboard] EMERGENCY STOP!");
        }

        #endregion

        #region Dashboard Updates

        private void UpdateDashboard()
        {
            if (_currentState == null) return;

            // Update gauges
            innerFrameGauge?.SetValue(_currentState.InnerVelocityRPM);
            outerFrameGauge?.SetValue(_currentState.OuterVelocityRPM);
            instantaneousGGauge?.SetValue(_currentState.instantaneousG);
            timeAveragedGGauge?.SetValue(_currentState.timeAveragedG);

            // Update text displays
            if (innerRPMText != null)
                innerRPMText.text = $"{_currentState.InnerVelocityRPM:F2} RPM";

            if (outerRPMText != null)
                outerRPMText.text = $"{_currentState.OuterVelocityRPM:F2} RPM";

            if (instantaneousGText != null)
                instantaneousGText.text = $"{_currentState.instantaneousG:F4} g";

            if (timeAveragedGText != null)
                timeAveragedGText.text = $"{_currentState.timeAveragedG:F5} g";

            // Update 3D visualization
            Update3DVisualization();
        }

        private void Update3DVisualization()
        {
            if (_currentState == null) return;

            // Rotate frame models
            if (innerFrameModel != null)
            {
                innerFrameModel.localRotation = Quaternion.Euler(
                    _currentState.InnerPositionDeg, 0f, 0f
                );
            }

            if (outerFrameModel != null)
            {
                outerFrameModel.localRotation = Quaternion.Euler(
                    0f, _currentState.OuterPositionDeg, 0f
                );
            }

            // Update gravity vector visualization
            if (gravityVectorLine != null)
            {
                Vector3 accel = _currentState.acceleration.ToUnityVector();
                
                // Scale for visualization
                Vector3 scaledAccel = accel.normalized * Mathf.Clamp(accel.magnitude / 9.81f, 0f, 1f);
                
                gravityVectorLine.SetPosition(0, Vector3.zero);
                gravityVectorLine.SetPosition(1, scaledAccel);
            }
        }

        private void UpdateConnectionStatus(ConnectionStatus status)
        {
            if (connectionStatusText != null)
            {
                connectionStatusText.text = status.ToString();
                connectionStatusText.color = status == ConnectionStatus.Connected 
                    ? connectedColor 
                    : disconnectedColor;
            }

            if (connectionIndicator != null)
            {
                connectionIndicator.color = status == ConnectionStatus.Connected 
                    ? connectedColor 
                    : disconnectedColor;
            }

            // Enable/disable controls based on connection
            bool connected = status == ConnectionStatus.Connected;
            
            if (innerRPMSlider != null)
                innerRPMSlider.interactable = connected;
            
            if (outerRPMSlider != null)
                outerRPMSlider.interactable = connected;
            
            if (startButton != null)
                startButton.interactable = connected;
            
            if (stopButton != null)
                stopButton.interactable = connected;
            
            // Emergency stop always available
            if (emergencyStopButton != null)
                emergencyStopButton.interactable = true;
        }

        private void UpdateMotorIndicators(bool innerRunning, bool outerRunning)
        {
            if (innerMotorIndicator != null)
                innerMotorIndicator.color = innerRunning ? runningColor : idleColor;

            if (outerMotorIndicator != null)
                outerMotorIndicator.color = outerRunning ? runningColor : idleColor;
        }

        private void UpdateTimestamp()
        {
            if (timestampText != null)
            {
                timestampText.text = DateTime.Now.ToString("HH:mm:ss.fff");
            }
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Set both frame speeds simultaneously
        /// </summary>
        public void SetBothFrameRPM(float innerRPM, float outerRPM)
        {
            if (innerRPMSlider != null)
                innerRPMSlider.value = innerRPM;
            
            if (outerRPMSlider != null)
                outerRPMSlider.value = outerRPM;
        }

        /// <summary>
        /// Reset dashboard to default state
        /// </summary>
        public void ResetDashboard()
        {
            SetBothFrameRPM(0f, 0f);
            _motorsRunning = false;
            UpdateMotorIndicators(false, false);
        }

        #endregion
    }
}
