/*
 * RPM Digital Twin - Arduino Firmware
 * ====================================
 * 
 * This firmware runs on Arduino (Mega 2560 or Due) and handles:
 * - IMU sensor reading (MPU9250 via I2C)
 * - Rotary encoder reading for both frames
 * - Motor control via step/direction signals
 * - Binary packet communication with host PC
 * 
 * Communication Protocol:
 * - Baud rate: 115200
 * - Packet format: [0xAA] [0x55] [TYPE] [LEN] [PAYLOAD...] [CRC16_H] [CRC16_L]
 * 
 * Copyright (c) 2024 RPM Digital Twin Team
 */

#include <Wire.h>

// =============================================================================
// Pin Definitions
// =============================================================================

// IMU (MPU9250) - I2C
#define MPU9250_ADDR 0x68

// Encoder pins (using interrupt-capable pins)
#define ENCODER_INNER_A 2    // Interrupt 0
#define ENCODER_INNER_B 3    // Interrupt 1
#define ENCODER_OUTER_A 18   // Interrupt 5
#define ENCODER_OUTER_B 19   // Interrupt 4

// Motor control pins
#define MOTOR_INNER_STEP 8
#define MOTOR_INNER_DIR 9
#define MOTOR_INNER_ENABLE 10

#define MOTOR_OUTER_STEP 11
#define MOTOR_OUTER_DIR 12
#define MOTOR_OUTER_ENABLE 13

// Status LED
#define LED_STATUS 13

// =============================================================================
// Protocol Constants
// =============================================================================

#define SYNC_BYTE_1 0xAA
#define SYNC_BYTE_2 0x55

// Packet types
#define PKT_TYPE_IMU_DATA       0x01
#define PKT_TYPE_ENCODER_DATA   0x02
#define PKT_TYPE_MOTOR_STATUS   0x03
#define PKT_TYPE_SYSTEM_STATUS  0x10
#define PKT_TYPE_MOTOR_COMMAND  0x20
#define PKT_TYPE_CONFIG         0x21
#define PKT_TYPE_ACK            0xF0
#define PKT_TYPE_ERROR          0xFF

// Motor states
#define MOTOR_STATE_IDLE      0
#define MOTOR_STATE_RUNNING   1
#define MOTOR_STATE_STOPPING  2
#define MOTOR_STATE_FAULT     3

// =============================================================================
// Configuration
// =============================================================================

// IMU configuration
#define IMU_SAMPLE_RATE_HZ 100
#define IMU_ACCEL_SCALE 16384.0f  // ±2g range
#define IMU_GYRO_SCALE 131.0f     // ±250 deg/s range

// Encoder configuration
#define ENCODER_PPR 2000  // Pulses per revolution

// Motor configuration
#define MOTOR_STEPS_PER_REV 200
#define MOTOR_MICROSTEP 16
#define STEPS_PER_REV (MOTOR_STEPS_PER_REV * MOTOR_MICROSTEP)

// Communication
#define SERIAL_BAUD 115200
#define TX_BUFFER_SIZE 64
#define RX_BUFFER_SIZE 64

// =============================================================================
// Global Variables
// =============================================================================

// IMU data
volatile float accel_x, accel_y, accel_z;
volatile float gyro_x, gyro_y, gyro_z;
volatile float mag_x, mag_y, mag_z;

// Encoder positions (volatile for interrupt safety)
volatile long encoder_inner_count = 0;
volatile long encoder_outer_count = 0;
volatile long encoder_inner_velocity = 0;  // counts per interval
volatile long encoder_outer_velocity = 0;

// Previous encoder values for velocity calculation
long prev_inner_count = 0;
long prev_outer_count = 0;
unsigned long prev_velocity_time = 0;

// Motor state
struct MotorState {
    uint8_t state;
    float target_rpm;
    float current_rpm;
    long step_count;
    unsigned long step_interval_us;
    bool direction;
    bool enabled;
} inner_motor, outer_motor;

// Timing
unsigned long last_imu_read = 0;
unsigned long last_encoder_send = 0;
unsigned long last_motor_step_inner = 0;
unsigned long last_motor_step_outer = 0;

// Communication buffers
uint8_t tx_buffer[TX_BUFFER_SIZE];
uint8_t rx_buffer[RX_BUFFER_SIZE];
uint8_t rx_index = 0;
bool receiving_packet = false;
uint8_t expected_length = 0;

// =============================================================================
// CRC16 Calculation
// =============================================================================

uint16_t crc16(uint8_t* data, uint8_t length) {
    uint16_t crc = 0xFFFF;
    
    for (uint8_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (uint8_t j = 0; j < 8; j++) {
            if (crc & 0x0001) {
                crc = (crc >> 1) ^ 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }
    
    return crc;
}

// =============================================================================
// Packet Transmission
// =============================================================================

void sendPacket(uint8_t type, uint8_t* payload, uint8_t payload_length) {
    // Build packet
    tx_buffer[0] = SYNC_BYTE_1;
    tx_buffer[1] = SYNC_BYTE_2;
    tx_buffer[2] = type;
    tx_buffer[3] = payload_length;
    
    // Copy payload
    for (uint8_t i = 0; i < payload_length; i++) {
        tx_buffer[4 + i] = payload[i];
    }
    
    // Calculate CRC over type, length, and payload
    uint16_t crc = crc16(&tx_buffer[2], payload_length + 2);
    tx_buffer[4 + payload_length] = (crc >> 8) & 0xFF;  // CRC high byte
    tx_buffer[5 + payload_length] = crc & 0xFF;          // CRC low byte
    
    // Send packet
    Serial.write(tx_buffer, 6 + payload_length);
}

// =============================================================================
// IMU Functions
// =============================================================================

void initIMU() {
    Wire.begin();
    Wire.setClock(400000);  // 400 kHz I2C
    
    // Wake up MPU9250
    Wire.beginTransmission(MPU9250_ADDR);
    Wire.write(0x6B);  // PWR_MGMT_1 register
    Wire.write(0x00);  // Wake up
    Wire.endTransmission(true);
    
    delay(100);
    
    // Configure accelerometer (±2g)
    Wire.beginTransmission(MPU9250_ADDR);
    Wire.write(0x1C);  // ACCEL_CONFIG
    Wire.write(0x00);  // ±2g
    Wire.endTransmission(true);
    
    // Configure gyroscope (±250 deg/s)
    Wire.beginTransmission(MPU9250_ADDR);
    Wire.write(0x1B);  // GYRO_CONFIG
    Wire.write(0x00);  // ±250 deg/s
    Wire.endTransmission(true);
}

void readIMU() {
    Wire.beginTransmission(MPU9250_ADDR);
    Wire.write(0x3B);  // Starting register (ACCEL_XOUT_H)
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU9250_ADDR, (uint8_t)14, (uint8_t)true);
    
    // Read accelerometer
    int16_t ax = (Wire.read() << 8) | Wire.read();
    int16_t ay = (Wire.read() << 8) | Wire.read();
    int16_t az = (Wire.read() << 8) | Wire.read();
    
    // Skip temperature
    Wire.read();
    Wire.read();
    
    // Read gyroscope
    int16_t gx = (Wire.read() << 8) | Wire.read();
    int16_t gy = (Wire.read() << 8) | Wire.read();
    int16_t gz = (Wire.read() << 8) | Wire.read();
    
    // Convert to physical units
    accel_x = ax / IMU_ACCEL_SCALE * 9.80665f;  // m/s²
    accel_y = ay / IMU_ACCEL_SCALE * 9.80665f;
    accel_z = az / IMU_ACCEL_SCALE * 9.80665f;
    
    gyro_x = gx / IMU_GYRO_SCALE;  // deg/s
    gyro_y = gy / IMU_GYRO_SCALE;
    gyro_z = gz / IMU_GYRO_SCALE;
}

void sendIMUData() {
    // Pack IMU data: 6 floats = 24 bytes + timestamp (4 bytes) = 28 bytes
    uint8_t payload[28];
    unsigned long timestamp = micros();
    
    // Timestamp (4 bytes, little-endian)
    payload[0] = timestamp & 0xFF;
    payload[1] = (timestamp >> 8) & 0xFF;
    payload[2] = (timestamp >> 16) & 0xFF;
    payload[3] = (timestamp >> 24) & 0xFF;
    
    // Accelerometer (3 x 4 bytes)
    memcpy(&payload[4], &accel_x, 4);
    memcpy(&payload[8], &accel_y, 4);
    memcpy(&payload[12], &accel_z, 4);
    
    // Gyroscope (3 x 4 bytes)
    memcpy(&payload[16], &gyro_x, 4);
    memcpy(&payload[20], &gyro_y, 4);
    memcpy(&payload[24], &gyro_z, 4);
    
    sendPacket(PKT_TYPE_IMU_DATA, payload, 28);
}

// =============================================================================
// Encoder Functions
// =============================================================================

void initEncoders() {
    // Inner encoder
    pinMode(ENCODER_INNER_A, INPUT_PULLUP);
    pinMode(ENCODER_INNER_B, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(ENCODER_INNER_A), encoderInnerISR, CHANGE);
    
    // Outer encoder
    pinMode(ENCODER_OUTER_A, INPUT_PULLUP);
    pinMode(ENCODER_OUTER_B, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(ENCODER_OUTER_A), encoderOuterISR, CHANGE);
}

void encoderInnerISR() {
    // Read both pins
    bool a = digitalRead(ENCODER_INNER_A);
    bool b = digitalRead(ENCODER_INNER_B);
    
    // Quadrature decoding
    if (a == b) {
        encoder_inner_count++;
    } else {
        encoder_inner_count--;
    }
}

void encoderOuterISR() {
    bool a = digitalRead(ENCODER_OUTER_A);
    bool b = digitalRead(ENCODER_OUTER_B);
    
    if (a == b) {
        encoder_outer_count++;
    } else {
        encoder_outer_count--;
    }
}

void calculateEncoderVelocity() {
    unsigned long now = micros();
    float dt = (now - prev_velocity_time) / 1000000.0f;  // seconds
    
    if (dt > 0.001) {  // At least 1ms
        noInterrupts();
        long inner_delta = encoder_inner_count - prev_inner_count;
        long outer_delta = encoder_outer_count - prev_outer_count;
        prev_inner_count = encoder_inner_count;
        prev_outer_count = encoder_outer_count;
        interrupts();
        
        encoder_inner_velocity = inner_delta / dt;  // counts per second
        encoder_outer_velocity = outer_delta / dt;
        
        prev_velocity_time = now;
    }
}

void sendEncoderData() {
    // Pack encoder data: 2 frames x (position + velocity) = 16 bytes + timestamp = 20 bytes
    uint8_t payload[20];
    unsigned long timestamp = micros();
    
    noInterrupts();
    long inner_pos = encoder_inner_count;
    long outer_pos = encoder_outer_count;
    interrupts();
    
    // Convert to radians
    float inner_rad = (float)inner_pos / (ENCODER_PPR * 4) * 2.0f * PI;  // x4 for quadrature
    float outer_rad = (float)outer_pos / (ENCODER_PPR * 4) * 2.0f * PI;
    
    // Velocity in rad/s
    float inner_vel = (float)encoder_inner_velocity / (ENCODER_PPR * 4) * 2.0f * PI;
    float outer_vel = (float)encoder_outer_velocity / (ENCODER_PPR * 4) * 2.0f * PI;
    
    // Timestamp
    payload[0] = timestamp & 0xFF;
    payload[1] = (timestamp >> 8) & 0xFF;
    payload[2] = (timestamp >> 16) & 0xFF;
    payload[3] = (timestamp >> 24) & 0xFF;
    
    // Inner encoder
    memcpy(&payload[4], &inner_rad, 4);
    memcpy(&payload[8], &inner_vel, 4);
    
    // Outer encoder
    memcpy(&payload[12], &outer_rad, 4);
    memcpy(&payload[16], &outer_vel, 4);
    
    sendPacket(PKT_TYPE_ENCODER_DATA, payload, 20);
}

// =============================================================================
// Motor Control Functions
// =============================================================================

void initMotors() {
    // Inner motor
    pinMode(MOTOR_INNER_STEP, OUTPUT);
    pinMode(MOTOR_INNER_DIR, OUTPUT);
    pinMode(MOTOR_INNER_ENABLE, OUTPUT);
    
    // Outer motor
    pinMode(MOTOR_OUTER_STEP, OUTPUT);
    pinMode(MOTOR_OUTER_DIR, OUTPUT);
    pinMode(MOTOR_OUTER_ENABLE, OUTPUT);
    
    // Initialize motor states
    inner_motor.state = MOTOR_STATE_IDLE;
    inner_motor.target_rpm = 0;
    inner_motor.current_rpm = 0;
    inner_motor.enabled = false;
    
    outer_motor.state = MOTOR_STATE_IDLE;
    outer_motor.target_rpm = 0;
    outer_motor.current_rpm = 0;
    outer_motor.enabled = false;
    
    // Disable motors initially
    digitalWrite(MOTOR_INNER_ENABLE, HIGH);  // Active LOW
    digitalWrite(MOTOR_OUTER_ENABLE, HIGH);
}

void setMotorRPM(MotorState* motor, float rpm) {
    motor->target_rpm = rpm;
    motor->direction = (rpm >= 0);
    
    float abs_rpm = abs(rpm);
    if (abs_rpm < 0.01) {
        motor->step_interval_us = 0;  // Stop
        motor->state = MOTOR_STATE_IDLE;
    } else {
        // Calculate step interval
        // steps/rev = STEPS_PER_REV
        // steps/sec = STEPS_PER_REV * rpm / 60
        // interval_us = 1000000 / steps_per_sec
        float steps_per_sec = (float)STEPS_PER_REV * abs_rpm / 60.0f;
        motor->step_interval_us = (unsigned long)(1000000.0f / steps_per_sec);
        motor->state = MOTOR_STATE_RUNNING;
    }
}

void updateMotors() {
    unsigned long now = micros();
    
    // Inner motor stepping
    if (inner_motor.state == MOTOR_STATE_RUNNING && inner_motor.step_interval_us > 0) {
        if (now - last_motor_step_inner >= inner_motor.step_interval_us) {
            digitalWrite(MOTOR_INNER_DIR, inner_motor.direction ? HIGH : LOW);
            digitalWrite(MOTOR_INNER_STEP, HIGH);
            delayMicroseconds(2);
            digitalWrite(MOTOR_INNER_STEP, LOW);
            last_motor_step_inner = now;
            inner_motor.step_count++;
        }
    }
    
    // Outer motor stepping
    if (outer_motor.state == MOTOR_STATE_RUNNING && outer_motor.step_interval_us > 0) {
        if (now - last_motor_step_outer >= outer_motor.step_interval_us) {
            digitalWrite(MOTOR_OUTER_DIR, outer_motor.direction ? HIGH : LOW);
            digitalWrite(MOTOR_OUTER_STEP, HIGH);
            delayMicroseconds(2);
            digitalWrite(MOTOR_OUTER_STEP, LOW);
            last_motor_step_outer = now;
            outer_motor.step_count++;
        }
    }
}

void enableMotor(MotorState* motor, uint8_t enable_pin, bool enable) {
    motor->enabled = enable;
    digitalWrite(enable_pin, enable ? LOW : HIGH);  // Active LOW
}

// =============================================================================
// Command Processing
// =============================================================================

void processPacket(uint8_t type, uint8_t* payload, uint8_t length) {
    switch (type) {
        case PKT_TYPE_MOTOR_COMMAND: {
            if (length >= 9) {
                uint8_t motor_id = payload[0];
                float rpm;
                memcpy(&rpm, &payload[1], 4);
                float acceleration;
                memcpy(&acceleration, &payload[5], 4);
                
                if (motor_id == 0) {  // Inner motor
                    enableMotor(&inner_motor, MOTOR_INNER_ENABLE, true);
                    setMotorRPM(&inner_motor, rpm);
                } else if (motor_id == 1) {  // Outer motor
                    enableMotor(&outer_motor, MOTOR_OUTER_ENABLE, true);
                    setMotorRPM(&outer_motor, rpm);
                }
                
                // Send ACK
                uint8_t ack_payload[] = {type, 0x00};  // Success
                sendPacket(PKT_TYPE_ACK, ack_payload, 2);
            }
            break;
        }
        
        case PKT_TYPE_CONFIG: {
            // Handle configuration commands
            uint8_t config_type = payload[0];
            
            if (config_type == 0x00) {  // Emergency stop
                inner_motor.state = MOTOR_STATE_IDLE;
                outer_motor.state = MOTOR_STATE_IDLE;
                enableMotor(&inner_motor, MOTOR_INNER_ENABLE, false);
                enableMotor(&outer_motor, MOTOR_OUTER_ENABLE, false);
                
                uint8_t ack_payload[] = {type, 0x00};
                sendPacket(PKT_TYPE_ACK, ack_payload, 2);
            }
            break;
        }
        
        default:
            // Unknown packet type
            uint8_t error_payload[] = {type, 0x01};  // Unknown command
            sendPacket(PKT_TYPE_ERROR, error_payload, 2);
            break;
    }
}

void handleSerialInput() {
    while (Serial.available()) {
        uint8_t byte = Serial.read();
        
        if (!receiving_packet) {
            // Look for sync bytes
            if (rx_index == 0 && byte == SYNC_BYTE_1) {
                rx_buffer[0] = byte;
                rx_index = 1;
            } else if (rx_index == 1 && byte == SYNC_BYTE_2) {
                rx_buffer[1] = byte;
                rx_index = 2;
                receiving_packet = true;
            } else {
                rx_index = 0;
            }
        } else {
            rx_buffer[rx_index++] = byte;
            
            // Check if we have type and length
            if (rx_index == 4) {
                expected_length = rx_buffer[3] + 6;  // header(4) + payload + crc(2)
            }
            
            // Check if packet is complete
            if (rx_index >= 4 && rx_index >= expected_length) {
                // Verify CRC
                uint16_t received_crc = (rx_buffer[rx_index - 2] << 8) | rx_buffer[rx_index - 1];
                uint16_t calculated_crc = crc16(&rx_buffer[2], rx_buffer[3] + 2);
                
                if (received_crc == calculated_crc) {
                    processPacket(rx_buffer[2], &rx_buffer[4], rx_buffer[3]);
                } else {
                    // CRC error
                    uint8_t error_payload[] = {0xFF, 0x02};  // CRC error
                    sendPacket(PKT_TYPE_ERROR, error_payload, 2);
                }
                
                // Reset for next packet
                rx_index = 0;
                receiving_packet = false;
            }
        }
        
        // Prevent buffer overflow
        if (rx_index >= RX_BUFFER_SIZE) {
            rx_index = 0;
            receiving_packet = false;
        }
    }
}

// =============================================================================
// System Status
// =============================================================================

void sendSystemStatus() {
    uint8_t payload[16];
    unsigned long uptime = millis();
    
    // Uptime (4 bytes)
    payload[0] = uptime & 0xFF;
    payload[1] = (uptime >> 8) & 0xFF;
    payload[2] = (uptime >> 16) & 0xFF;
    payload[3] = (uptime >> 24) & 0xFF;
    
    // Motor states
    payload[4] = inner_motor.state;
    payload[5] = outer_motor.state;
    
    // Motor RPMs (as int16)
    int16_t inner_rpm = (int16_t)(inner_motor.current_rpm * 10);
    int16_t outer_rpm = (int16_t)(outer_motor.current_rpm * 10);
    memcpy(&payload[6], &inner_rpm, 2);
    memcpy(&payload[8], &outer_rpm, 2);
    
    // System flags
    payload[10] = (inner_motor.enabled ? 0x01 : 0x00) | 
                  (outer_motor.enabled ? 0x02 : 0x00);
    
    // Reserved
    payload[11] = 0;
    payload[12] = 0;
    payload[13] = 0;
    payload[14] = 0;
    payload[15] = 0;
    
    sendPacket(PKT_TYPE_SYSTEM_STATUS, payload, 16);
}

// =============================================================================
// Setup and Main Loop
// =============================================================================

void setup() {
    // Initialize serial
    Serial.begin(SERIAL_BAUD);
    while (!Serial && millis() < 3000);  // Wait up to 3 seconds for serial
    
    // Initialize status LED
    pinMode(LED_STATUS, OUTPUT);
    digitalWrite(LED_STATUS, HIGH);
    
    // Initialize subsystems
    initIMU();
    initEncoders();
    initMotors();
    
    // Initialize timing
    prev_velocity_time = micros();
    
    // Send startup message
    uint8_t startup_payload[] = {0x01, 0x00, 0x00};  // Version 1.0.0
    sendPacket(PKT_TYPE_SYSTEM_STATUS, startup_payload, 3);
    
    digitalWrite(LED_STATUS, LOW);
}

void loop() {
    unsigned long now = millis();
    
    // Handle incoming commands
    handleSerialInput();
    
    // Update motor stepping (high frequency)
    updateMotors();
    
    // Read and send IMU data at specified rate
    if (now - last_imu_read >= (1000 / IMU_SAMPLE_RATE_HZ)) {
        readIMU();
        sendIMUData();
        last_imu_read = now;
    }
    
    // Calculate encoder velocity and send data
    calculateEncoderVelocity();
    if (now - last_encoder_send >= 20) {  // 50 Hz
        sendEncoderData();
        last_encoder_send = now;
    }
    
    // Blink status LED
    digitalWrite(LED_STATUS, (now / 500) % 2);
}
