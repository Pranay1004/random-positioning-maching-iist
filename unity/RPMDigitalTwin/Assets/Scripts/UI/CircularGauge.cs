/*
 * RPM Digital Twin - Circular Gauge Component
 * ============================================
 * 
 * Tesla-style animated circular gauge for displaying values.
 * 
 * Features:
 * - Smooth value animation
 * - Configurable range and colors
 * - Glow effects
 * - Value labels
 */

using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace RPMDigitalTwin.UI
{
    /// <summary>
    /// Animated circular gauge component
    /// </summary>
    public class CircularGauge : MonoBehaviour
    {
        #region Inspector Fields

        [Header("Gauge Settings")]
        [SerializeField] private float minValue = 0f;
        [SerializeField] private float maxValue = 100f;
        [SerializeField] private float currentValue = 0f;
        [SerializeField] private string unit = "";
        [SerializeField] private string formatString = "F1";

        [Header("Animation")]
        [SerializeField] private float animationSpeed = 5f;
        [SerializeField] private AnimationCurve animationCurve = AnimationCurve.EaseInOut(0f, 0f, 1f, 1f);

        [Header("Visual References")]
        [SerializeField] private Image fillImage;
        [SerializeField] private Image glowImage;
        [SerializeField] private TextMeshProUGUI valueText;
        [SerializeField] private TextMeshProUGUI labelText;
        [SerializeField] private RectTransform needle;

        [Header("Colors")]
        [SerializeField] private Gradient fillGradient;
        [SerializeField] private Color normalColor = new Color(0f, 0.8f, 1f);
        [SerializeField] private Color warningColor = new Color(1f, 0.7f, 0f);
        [SerializeField] private Color dangerColor = new Color(1f, 0.2f, 0.2f);
        [SerializeField] private float warningThreshold = 0.7f;
        [SerializeField] private float dangerThreshold = 0.9f;

        [Header("Glow Settings")]
        [SerializeField] private float glowIntensity = 1.5f;
        [SerializeField] private float glowPulseSpeed = 2f;
        [SerializeField] private bool enableGlowPulse = true;

        #endregion

        #region Private Fields

        private float _displayValue;
        private float _targetValue;
        private float _fillAmount;
        private float _needleAngle;
        private bool _isAnimating;

        // Angle range for the gauge (typically 270 degrees)
        private const float MIN_ANGLE = 135f;
        private const float MAX_ANGLE = -135f;
        private const float ANGLE_RANGE = 270f;

        #endregion

        #region Properties

        /// <summary>
        /// Current gauge value
        /// </summary>
        public float Value
        {
            get => currentValue;
            set => SetValue(value);
        }

        /// <summary>
        /// Normalized value (0-1)
        /// </summary>
        public float NormalizedValue => Mathf.InverseLerp(minValue, maxValue, currentValue);

        #endregion

        #region Unity Lifecycle

        private void Awake()
        {
            InitializeGauge();
        }

        private void Update()
        {
            AnimateGauge();
            UpdateGlow();
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Set the gauge value with animation
        /// </summary>
        public void SetValue(float value)
        {
            _targetValue = Mathf.Clamp(value, minValue, maxValue);
            currentValue = _targetValue;
            _isAnimating = true;
        }

        /// <summary>
        /// Set value immediately without animation
        /// </summary>
        public void SetValueImmediate(float value)
        {
            _targetValue = Mathf.Clamp(value, minValue, maxValue);
            currentValue = _targetValue;
            _displayValue = _targetValue;
            UpdateGaugeVisuals();
        }

        /// <summary>
        /// Set the gauge range
        /// </summary>
        public void SetRange(float min, float max)
        {
            minValue = min;
            maxValue = max;
            UpdateGaugeVisuals();
        }

        /// <summary>
        /// Set the gauge label
        /// </summary>
        public void SetLabel(string label)
        {
            if (labelText != null)
            {
                labelText.text = label;
            }
        }

        /// <summary>
        /// Set the unit string
        /// </summary>
        public void SetUnit(string newUnit)
        {
            unit = newUnit;
            UpdateValueText();
        }

        #endregion

        #region Private Methods

        private void InitializeGauge()
        {
            _displayValue = currentValue;
            _targetValue = currentValue;
            
            // Initialize fill gradient if not set
            if (fillGradient == null)
            {
                fillGradient = new Gradient();
                var colorKeys = new GradientColorKey[]
                {
                    new GradientColorKey(normalColor, 0f),
                    new GradientColorKey(normalColor, warningThreshold),
                    new GradientColorKey(warningColor, warningThreshold + 0.01f),
                    new GradientColorKey(dangerColor, dangerThreshold)
                };
                var alphaKeys = new GradientAlphaKey[]
                {
                    new GradientAlphaKey(1f, 0f),
                    new GradientAlphaKey(1f, 1f)
                };
                fillGradient.SetKeys(colorKeys, alphaKeys);
            }

            UpdateGaugeVisuals();
        }

        private void AnimateGauge()
        {
            if (!_isAnimating) return;

            // Smooth animation towards target
            float delta = _targetValue - _displayValue;
            float step = delta * animationSpeed * Time.deltaTime;

            if (Mathf.Abs(delta) < 0.001f)
            {
                _displayValue = _targetValue;
                _isAnimating = false;
            }
            else
            {
                _displayValue += step;
            }

            UpdateGaugeVisuals();
        }

        private void UpdateGaugeVisuals()
        {
            // Calculate normalized value
            float normalizedValue = Mathf.InverseLerp(minValue, maxValue, _displayValue);
            
            // Update fill
            if (fillImage != null)
            {
                fillImage.fillAmount = normalizedValue;
                fillImage.color = fillGradient.Evaluate(normalizedValue);
            }

            // Update needle
            if (needle != null)
            {
                float angle = Mathf.Lerp(MIN_ANGLE, MAX_ANGLE, normalizedValue);
                needle.localRotation = Quaternion.Euler(0f, 0f, angle);
            }

            // Update glow color
            if (glowImage != null)
            {
                Color glowColor = fillGradient.Evaluate(normalizedValue);
                glowColor.a = 0.5f;
                glowImage.color = glowColor;
            }

            // Update text
            UpdateValueText();
        }

        private void UpdateValueText()
        {
            if (valueText != null)
            {
                string valueStr = _displayValue.ToString(formatString);
                valueText.text = string.IsNullOrEmpty(unit) ? valueStr : $"{valueStr} {unit}";
            }
        }

        private void UpdateGlow()
        {
            if (!enableGlowPulse || glowImage == null) return;

            // Pulse glow based on value
            float normalizedValue = NormalizedValue;
            float baseAlpha = 0.3f + normalizedValue * 0.4f;
            float pulse = Mathf.Sin(Time.time * glowPulseSpeed) * 0.15f * normalizedValue;
            
            Color glowColor = glowImage.color;
            glowColor.a = baseAlpha + pulse;
            glowImage.color = glowColor;
        }

        #endregion

        #region Editor Support

        private void OnValidate()
        {
            if (Application.isPlaying)
            {
                SetValue(currentValue);
            }
            else
            {
                InitializeGauge();
            }
        }

        #endregion
    }
}
