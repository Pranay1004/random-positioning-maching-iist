%% RPM Digital Twin - MATLAB Main Entry Point
%% ============================================
%% This script initializes the MATLAB components of the RPM Digital Twin system.
%% 
%% MATLAB is used for:
%% - Scientific computing core
%% - Simulink real-time simulation
%% - HIL/SIL validation
%% - Advanced numerical analysis
%%
%% Copyright (c) 2024 RPM Digital Twin Team

%% Clear workspace
clear; clc; close all;

%% Add paths
fprintf('RPM Digital Twin - MATLAB Engine\n');
fprintf('================================\n\n');

% Get the directory where this script is located
thisDir = fileparts(mfilename('fullpath'));
projectRoot = fullfile(thisDir, '..', '..');

% Add all MATLAB paths
addpath(genpath(fullfile(thisDir, 'functions')));
addpath(genpath(fullfile(thisDir, 'models')));
addpath(genpath(fullfile(thisDir, 'scripts')));

fprintf('MATLAB paths configured.\n');

%% Load configuration
configFile = fullfile(projectRoot, 'config', 'main_config.yaml');
if exist(configFile, 'file')
    fprintf('Configuration file found.\n');
else
    warning('Configuration file not found: %s', configFile);
end

%% Initialize RPM parameters
fprintf('\nInitializing RPM parameters...\n');

% Physical constants
g = 9.80665;  % Earth gravity (m/s^2)

% RPM Geometry
rpm_params = struct();
rpm_params.inner_frame_radius = 0.15;  % m
rpm_params.outer_frame_radius = 0.25;  % m
rpm_params.sample_offset = [0; 0; 0.10];  % m

% Motor parameters
rpm_params.motor_max_rpm = 60;  % Maximum rotational speed
rpm_params.motor_acceleration = 100;  % RPM/s

fprintf('  Inner frame radius: %.3f m\n', rpm_params.inner_frame_radius);
fprintf('  Outer frame radius: %.3f m\n', rpm_params.outer_frame_radius);
fprintf('  Sample position: [%.3f, %.3f, %.3f] m\n', rpm_params.sample_offset);

%% Display status
fprintf('\n');
fprintf('RPM Digital Twin MATLAB Engine Ready\n');
fprintf('=====================================\n');
fprintf('  MATLAB Version: %s\n', version);
fprintf('  Platform: %s\n', computer);
fprintf('  Date: %s\n', datestr(now));
fprintf('\n');
fprintf('Available functions:\n');
fprintf('  - compute_gravity_vector(theta_i, theta_o)\n');
fprintf('  - compute_rotation_matrix(theta_i, theta_o)\n');
fprintf('  - simulate_rpm(duration, dt, omega_i, omega_o)\n');
fprintf('  - analyze_microgravity(position, omega_i, omega_o)\n');
fprintf('\n');

%% Store in base workspace for easy access
assignin('base', 'rpm_params', rpm_params);
assignin('base', 'g', g);

fprintf('Parameters stored in workspace as ''rpm_params'' and ''g''.\n');
