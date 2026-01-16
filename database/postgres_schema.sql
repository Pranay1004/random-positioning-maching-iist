-- RPM Digital Twin - PostgreSQL Database Schema
-- ==============================================
-- Metadata storage for experiments, configurations, and users.
-- Time-series data is stored in InfluxDB.
--
-- Target: PostgreSQL 14+
-- Copyright (c) 2024 RPM Digital Twin Team

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ====================
-- SCHEMA CREATION
-- ====================

CREATE SCHEMA IF NOT EXISTS rpm_twin;
SET search_path TO rpm_twin, public;

-- ====================
-- ENUMS
-- ====================

CREATE TYPE experiment_status AS ENUM (
    'planned',
    'running',
    'paused',
    'completed',
    'aborted',
    'failed'
);

CREATE TYPE operation_mode AS ENUM (
    'clinostat_3d',
    'random',
    'partial_gravity',
    'custom'
);

CREATE TYPE user_role AS ENUM (
    'admin',
    'researcher',
    'operator',
    'viewer'
);

-- ====================
-- TABLES
-- ====================

-- Users and authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    organization VARCHAR(255),
    role user_role DEFAULT 'viewer',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

-- RPM device configurations
CREATE TABLE devices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    serial_number VARCHAR(50) UNIQUE,
    model VARCHAR(100),
    firmware_version VARCHAR(50),
    
    -- Geometry parameters
    inner_frame_radius DECIMAL(8,5) NOT NULL DEFAULT 0.15,
    outer_frame_radius DECIMAL(8,5) NOT NULL DEFAULT 0.25,
    max_inner_rpm DECIMAL(8,3) DEFAULT 60.0,
    max_outer_rpm DECIMAL(8,3) DEFAULT 60.0,
    
    -- Calibration data (JSON)
    calibration_data JSONB,
    
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Experiment definitions
CREATE TABLE experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Foreign keys
    device_id UUID REFERENCES devices(id),
    created_by UUID REFERENCES users(id),
    
    -- Experiment parameters
    operation_mode operation_mode NOT NULL,
    inner_rpm_setpoint DECIMAL(8,3),
    outer_rpm_setpoint DECIMAL(8,3),
    target_g DECIMAL(8,6),
    duration_planned_seconds INTEGER,
    
    -- Sample information
    sample_type VARCHAR(100),
    sample_description TEXT,
    sample_position_x DECIMAL(8,5) DEFAULT 0,
    sample_position_y DECIMAL(8,5) DEFAULT 0,
    sample_position_z DECIMAL(8,5) DEFAULT 0.1,
    
    -- Execution details
    status experiment_status DEFAULT 'planned',
    started_at TIMESTAMP WITH TIME ZONE,
    ended_at TIMESTAMP WITH TIME ZONE,
    duration_actual_seconds INTEGER,
    
    -- Results summary (computed)
    mean_g DECIMAL(10,8),
    std_g DECIMAL(10,8),
    max_g DECIMAL(10,8),
    min_g DECIMAL(10,8),
    data_quality_score DECIMAL(5,2),
    
    -- Metadata
    tags VARCHAR(50)[],
    notes TEXT,
    custom_metadata JSONB,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Experiment events/log
CREATE TABLE experiment_events (
    id BIGSERIAL PRIMARY KEY,
    experiment_id UUID NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Device configuration history
CREATE TABLE device_config_history (
    id BIGSERIAL PRIMARY KEY,
    device_id UUID NOT NULL REFERENCES devices(id) ON DELETE CASCADE,
    config_snapshot JSONB NOT NULL,
    changed_by UUID REFERENCES users(id),
    change_reason TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Calibration records
CREATE TABLE calibration_records (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    device_id UUID NOT NULL REFERENCES devices(id),
    performed_by UUID REFERENCES users(id),
    
    -- Calibration type
    calibration_type VARCHAR(50) NOT NULL, -- 'imu', 'encoder', 'motor', 'full'
    
    -- Results
    status VARCHAR(20) NOT NULL, -- 'passed', 'failed', 'warning'
    results JSONB NOT NULL,
    
    -- Metadata
    notes TEXT,
    performed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Preset configurations for common experiment types
CREATE TABLE experiment_presets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- Preset parameters
    operation_mode operation_mode NOT NULL,
    inner_rpm_setpoint DECIMAL(8,3),
    outer_rpm_setpoint DECIMAL(8,3),
    target_g DECIMAL(8,6),
    default_duration_seconds INTEGER,
    
    -- Sample position defaults
    sample_position DECIMAL(8,5)[3],
    
    -- Additional settings
    settings JSONB,
    
    is_public BOOLEAN DEFAULT false,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Simulation runs (for digital twin validation)
CREATE TABLE simulation_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments(id),
    
    -- Simulation parameters
    duration_seconds DECIMAL(10,3) NOT NULL,
    time_step_ms DECIMAL(8,4) DEFAULT 10.0,
    operation_mode operation_mode NOT NULL,
    inner_rpm DECIMAL(8,3),
    outer_rpm DECIMAL(8,3),
    sample_position DECIMAL(8,5)[3],
    
    -- Results
    mean_g DECIMAL(10,8),
    std_g DECIMAL(10,8),
    max_g DECIMAL(10,8),
    min_g DECIMAL(10,8),
    
    -- Comparison with experiment (if linked)
    experiment_deviation_mean_g DECIMAL(10,8),
    experiment_deviation_std_g DECIMAL(10,8),
    validation_passed BOOLEAN,
    
    -- Metadata
    ran_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ran_by UUID REFERENCES users(id),
    computation_time_ms INTEGER,
    notes TEXT
);

-- ====================
-- INDEXES
-- ====================

CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_experiments_device ON experiments(device_id);
CREATE INDEX idx_experiments_created_by ON experiments(created_by);
CREATE INDEX idx_experiments_started_at ON experiments(started_at);
CREATE INDEX idx_experiments_tags ON experiments USING GIN(tags);

CREATE INDEX idx_experiment_events_experiment ON experiment_events(experiment_id);
CREATE INDEX idx_experiment_events_type ON experiment_events(event_type);
CREATE INDEX idx_experiment_events_timestamp ON experiment_events(timestamp);

CREATE INDEX idx_calibration_device ON calibration_records(device_id);
CREATE INDEX idx_calibration_type ON calibration_records(calibration_type);

CREATE INDEX idx_simulation_experiment ON simulation_runs(experiment_id);
CREATE INDEX idx_simulation_ran_at ON simulation_runs(ran_at);

-- ====================
-- FUNCTIONS
-- ====================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Calculate experiment duration
CREATE OR REPLACE FUNCTION calculate_experiment_duration()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.ended_at IS NOT NULL AND NEW.started_at IS NOT NULL THEN
        NEW.duration_actual_seconds = EXTRACT(EPOCH FROM (NEW.ended_at - NEW.started_at))::INTEGER;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ====================
-- TRIGGERS
-- ====================

CREATE TRIGGER update_users_timestamp
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_devices_timestamp
    BEFORE UPDATE ON devices
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_experiments_timestamp
    BEFORE UPDATE ON experiments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER calculate_experiment_duration_trigger
    BEFORE UPDATE OF ended_at ON experiments
    FOR EACH ROW
    EXECUTE FUNCTION calculate_experiment_duration();

-- ====================
-- VIEWS
-- ====================

-- Recent experiments view
CREATE VIEW recent_experiments AS
SELECT 
    e.id,
    e.name,
    e.status,
    e.operation_mode,
    e.started_at,
    e.ended_at,
    e.duration_actual_seconds,
    e.mean_g,
    e.std_g,
    d.name as device_name,
    u.email as created_by_email,
    u.first_name || ' ' || u.last_name as created_by_name
FROM experiments e
LEFT JOIN devices d ON e.device_id = d.id
LEFT JOIN users u ON e.created_by = u.id
ORDER BY e.created_at DESC;

-- Device status view
CREATE VIEW device_status AS
SELECT 
    d.id,
    d.name,
    d.serial_number,
    d.is_active,
    COUNT(e.id) FILTER (WHERE e.status = 'running') as running_experiments,
    COUNT(e.id) as total_experiments,
    MAX(e.ended_at) as last_experiment_ended
FROM devices d
LEFT JOIN experiments e ON d.id = e.device_id
GROUP BY d.id, d.name, d.serial_number, d.is_active;

-- Experiment statistics view
CREATE VIEW experiment_statistics AS
SELECT 
    operation_mode,
    COUNT(*) as total_count,
    COUNT(*) FILTER (WHERE status = 'completed') as completed_count,
    AVG(duration_actual_seconds) as avg_duration_seconds,
    AVG(mean_g) as avg_mean_g,
    AVG(std_g) as avg_std_g,
    MIN(mean_g) as best_mean_g
FROM experiments
WHERE status = 'completed'
GROUP BY operation_mode;

-- ====================
-- INITIAL DATA
-- ====================

-- Default device
INSERT INTO devices (name, serial_number, model, firmware_version)
VALUES ('RPM-001', 'RPM2024001', 'RPM Digital Twin v1', '1.0.0');

-- Default presets
INSERT INTO experiment_presets (name, description, operation_mode, inner_rpm_setpoint, outer_rpm_setpoint, target_g, default_duration_seconds, sample_position)
VALUES 
    ('3D Clinostat Standard', 'Standard 3D clinostat mode with equal rotation speeds', 'clinostat_3d', 2.0, 2.0, 0.01, 3600, ARRAY[0.0, 0.0, 0.1]),
    ('Random Mode High Speed', 'Random direction changes at higher speeds', 'random', 5.0, 5.0, 0.02, 1800, ARRAY[0.0, 0.0, 0.1]),
    ('Lunar Gravity', 'Simulate lunar gravity (~0.166g)', 'partial_gravity', 1.0, 0.0, 0.166, 7200, ARRAY[0.0, 0.0, 0.1]),
    ('Mars Gravity', 'Simulate Mars gravity (~0.38g)', 'partial_gravity', 0.8, 0.0, 0.38, 7200, ARRAY[0.0, 0.0, 0.1]);

-- ====================
-- PERMISSIONS
-- ====================

-- Create application role
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rpm_app') THEN
        CREATE ROLE rpm_app WITH LOGIN PASSWORD 'change_in_production';
    END IF;
END
$$;

GRANT USAGE ON SCHEMA rpm_twin TO rpm_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA rpm_twin TO rpm_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA rpm_twin TO rpm_app;

-- Grant access to views
GRANT SELECT ON recent_experiments TO rpm_app;
GRANT SELECT ON device_status TO rpm_app;
GRANT SELECT ON experiment_statistics TO rpm_app;

COMMENT ON SCHEMA rpm_twin IS 'RPM Digital Twin metadata storage';
