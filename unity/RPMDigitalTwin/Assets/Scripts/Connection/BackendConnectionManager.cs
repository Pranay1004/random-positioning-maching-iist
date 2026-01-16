/*
 * RPM Digital Twin - Unity UI Connection Manager
 * ===============================================
 * 
 * Handles communication between Unity UI and Python backend via gRPC.
 * 
 * This script should be attached to a persistent GameObject in the Unity scene.
 * 
 * Features:
 * - Async connection management
 * - Real-time state streaming
 * - Motor command sending
 * - Automatic reconnection
 */

using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Events;

namespace RPMDigitalTwin
{
    /// <summary>
    /// 3D Vector data from backend
    /// </summary>
    [Serializable]
    public struct Vector3Data
    {
        public float x;
        public float y;
        public float z;

        public Vector3 ToUnityVector()
        {
            return new Vector3(x, y, z);
        }
    }

    /// <summary>
    /// RPM state update from backend
    /// </summary>
    [Serializable]
    public class RPMStateData
    {
        public long timestampMs;
        public float innerPositionRad;
        public float innerVelocityRadS;
        public float outerPositionRad;
        public float outerVelocityRadS;
        public Vector3Data acceleration;
        public float instantaneousG;
        public float timeAveragedG;

        /// <summary>
        /// Inner frame position in degrees
        /// </summary>
        public float InnerPositionDeg => innerPositionRad * Mathf.Rad2Deg;

        /// <summary>
        /// Outer frame position in degrees
        /// </summary>
        public float OuterPositionDeg => outerPositionRad * Mathf.Rad2Deg;

        /// <summary>
        /// Inner frame velocity in RPM
        /// </summary>
        public float InnerVelocityRPM => innerVelocityRadS * 60f / (2f * Mathf.PI);

        /// <summary>
        /// Outer frame velocity in RPM
        /// </summary>
        public float OuterVelocityRPM => outerVelocityRadS * 60f / (2f * Mathf.PI);
    }

    /// <summary>
    /// Motor command to send to backend
    /// </summary>
    [Serializable]
    public class MotorCommand
    {
        public int motorId; // 0 = inner, 1 = outer
        public float rpm;
        public float acceleration;

        public MotorCommand(int id, float targetRpm, float accel = 100f)
        {
            motorId = id;
            rpm = targetRpm;
            acceleration = accel;
        }
    }

    /// <summary>
    /// Connection status enum
    /// </summary>
    public enum ConnectionStatus
    {
        Disconnected,
        Connecting,
        Connected,
        Reconnecting,
        Error
    }

    /// <summary>
    /// Unity Events for state updates
    /// </summary>
    [Serializable]
    public class RPMStateEvent : UnityEvent<RPMStateData> { }

    [Serializable]
    public class ConnectionStatusEvent : UnityEvent<ConnectionStatus> { }

    /// <summary>
    /// Main connection manager for RPM Digital Twin backend
    /// </summary>
    public class BackendConnectionManager : MonoBehaviour
    {
        #region Inspector Fields

        [Header("Connection Settings")]
        [SerializeField] private string serverHost = "localhost";
        [SerializeField] private int serverPort = 50051;
        [SerializeField] private bool autoConnect = true;
        [SerializeField] private float reconnectDelaySeconds = 3f;

        [Header("Events")]
        public RPMStateEvent OnStateUpdate;
        public ConnectionStatusEvent OnConnectionStatusChanged;

        #endregion

        #region Properties

        /// <summary>
        /// Current connection status
        /// </summary>
        public ConnectionStatus Status { get; private set; } = ConnectionStatus.Disconnected;

        /// <summary>
        /// Latest received state
        /// </summary>
        public RPMStateData CurrentState { get; private set; }

        /// <summary>
        /// Whether connected to backend
        /// </summary>
        public bool IsConnected => Status == ConnectionStatus.Connected;

        #endregion

        #region Private Fields

        private CancellationTokenSource _cancellationSource;
        private bool _shouldReconnect = true;
        private Queue<RPMStateData> _stateQueue = new Queue<RPMStateData>();
        private Queue<MotorCommand> _commandQueue = new Queue<MotorCommand>();
        private readonly object _stateLock = new object();
        private readonly object _commandLock = new object();

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            // Initialize events if null
            OnStateUpdate ??= new RPMStateEvent();
            OnConnectionStatusChanged ??= new ConnectionStatusEvent();
        }

        private void Start()
        {
            if (autoConnect)
            {
                Connect();
            }
        }

        private void Update()
        {
            // Process state updates on main thread
            ProcessStateQueue();
        }

        private void OnDestroy()
        {
            Disconnect();
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Connect to the backend server
        /// </summary>
        public void Connect()
        {
            if (Status == ConnectionStatus.Connected || Status == ConnectionStatus.Connecting)
            {
                Debug.LogWarning("[RPM] Already connected or connecting");
                return;
            }

            _cancellationSource = new CancellationTokenSource();
            _shouldReconnect = true;

            StartCoroutine(ConnectionCoroutine());
        }

        /// <summary>
        /// Disconnect from the backend server
        /// </summary>
        public void Disconnect()
        {
            _shouldReconnect = false;
            _cancellationSource?.Cancel();
            
            SetConnectionStatus(ConnectionStatus.Disconnected);
            Debug.Log("[RPM] Disconnected from backend");
        }

        /// <summary>
        /// Send motor command to backend
        /// </summary>
        /// <param name="motorId">0 for inner, 1 for outer</param>
        /// <param name="rpm">Target RPM</param>
        /// <param name="acceleration">Acceleration in RPM/s</param>
        public void SendMotorCommand(int motorId, float rpm, float acceleration = 100f)
        {
            var command = new MotorCommand(motorId, rpm, acceleration);
            
            lock (_commandLock)
            {
                _commandQueue.Enqueue(command);
            }

            Debug.Log($"[RPM] Motor command queued: Motor {motorId}, RPM {rpm}");
        }

        /// <summary>
        /// Set inner frame speed
        /// </summary>
        public void SetInnerFrameRPM(float rpm) => SendMotorCommand(0, rpm);

        /// <summary>
        /// Set outer frame speed
        /// </summary>
        public void SetOuterFrameRPM(float rpm) => SendMotorCommand(1, rpm);

        /// <summary>
        /// Emergency stop both motors
        /// </summary>
        public void EmergencyStop()
        {
            SendMotorCommand(0, 0, 1000f); // High acceleration for quick stop
            SendMotorCommand(1, 0, 1000f);
            Debug.LogWarning("[RPM] EMERGENCY STOP");
        }

        #endregion

        #region Private Methods

        private IEnumerator ConnectionCoroutine()
        {
            SetConnectionStatus(ConnectionStatus.Connecting);

            while (_shouldReconnect && !_cancellationSource.IsCancellationRequested)
            {
                // Attempt connection
                bool connected = false;
                
                // Simulated connection - in real implementation, use gRPC
                yield return new WaitForSeconds(0.5f);
                
                // For demo, assume connection succeeds
                connected = true;

                if (connected)
                {
                    SetConnectionStatus(ConnectionStatus.Connected);
                    Debug.Log($"[RPM] Connected to backend at {serverHost}:{serverPort}");

                    // Start receiving data
                    StartCoroutine(ReceiveDataCoroutine());
                    StartCoroutine(SendCommandsCoroutine());

                    // Wait until disconnected
                    while (Status == ConnectionStatus.Connected && 
                           !_cancellationSource.IsCancellationRequested)
                    {
                        yield return null;
                    }
                }
                else
                {
                    Debug.LogWarning("[RPM] Connection failed, will retry...");
                }

                if (_shouldReconnect && !_cancellationSource.IsCancellationRequested)
                {
                    SetConnectionStatus(ConnectionStatus.Reconnecting);
                    yield return new WaitForSeconds(reconnectDelaySeconds);
                }
            }
        }

        private IEnumerator ReceiveDataCoroutine()
        {
            // Simulated data reception - in real implementation, stream from gRPC
            while (Status == ConnectionStatus.Connected && 
                   !_cancellationSource.IsCancellationRequested)
            {
                // Generate mock data for testing
                var state = new RPMStateData
                {
                    timestampMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                    innerPositionRad = Time.time * 2f * Mathf.PI / 60f, // Simulated rotation
                    innerVelocityRadS = 2f * Mathf.PI / 60f, // 1 RPM
                    outerPositionRad = Time.time * 2f * Mathf.PI / 30f,
                    outerVelocityRadS = 2f * Mathf.PI / 30f, // 2 RPM
                    acceleration = new Vector3Data { x = 0.1f, y = -9.8f, z = 0.05f },
                    instantaneousG = 0.02f + 0.01f * Mathf.Sin(Time.time * 5f),
                    timeAveragedG = 0.015f
                };

                lock (_stateLock)
                {
                    _stateQueue.Enqueue(state);
                    
                    // Limit queue size
                    while (_stateQueue.Count > 10)
                    {
                        _stateQueue.Dequeue();
                    }
                }

                yield return new WaitForSeconds(0.01f); // 100 Hz update rate
            }
        }

        private IEnumerator SendCommandsCoroutine()
        {
            while (Status == ConnectionStatus.Connected && 
                   !_cancellationSource.IsCancellationRequested)
            {
                MotorCommand command = null;
                
                lock (_commandLock)
                {
                    if (_commandQueue.Count > 0)
                    {
                        command = _commandQueue.Dequeue();
                    }
                }

                if (command != null)
                {
                    // In real implementation, send via gRPC
                    Debug.Log($"[RPM] Sending command: Motor {command.motorId}, RPM {command.rpm}");
                }

                yield return new WaitForSeconds(0.01f);
            }
        }

        private void ProcessStateQueue()
        {
            RPMStateData state = null;
            
            lock (_stateLock)
            {
                if (_stateQueue.Count > 0)
                {
                    state = _stateQueue.Dequeue();
                }
            }

            if (state != null)
            {
                CurrentState = state;
                OnStateUpdate?.Invoke(state);
            }
        }

        private void SetConnectionStatus(ConnectionStatus newStatus)
        {
            if (Status != newStatus)
            {
                Status = newStatus;
                OnConnectionStatusChanged?.Invoke(newStatus);
            }
        }

        #endregion
    }
}
