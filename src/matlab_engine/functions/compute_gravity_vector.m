function [g_vector, g_magnitude] = compute_gravity_vector(theta_inner, theta_outer, position, omega_inner, omega_outer)
%COMPUTE_GRAVITY_VECTOR Compute effective gravity at a point on RPM
%
%   [g_vector, g_magnitude] = COMPUTE_GRAVITY_VECTOR(theta_inner, theta_outer)
%   computes the effective gravity vector experienced at the sample position
%   in the rotating frame of the Random Positioning Machine.
%
%   [g_vector, g_magnitude] = COMPUTE_GRAVITY_VECTOR(theta_inner, theta_outer, position)
%   computes gravity at a specific position.
%
%   [g_vector, g_magnitude] = COMPUTE_GRAVITY_VECTOR(theta_inner, theta_outer, position, omega_inner, omega_outer)
%   includes centrifugal acceleration effects.
%
%   Inputs:
%       theta_inner - Inner frame angle (radians)
%       theta_outer - Outer frame angle (radians)
%       position    - 3x1 position vector in sample frame [x;y;z] (m)
%                     Default: [0; 0; 0.1]
%       omega_inner - Inner frame angular velocity (rad/s), default: 0
%       omega_outer - Outer frame angular velocity (rad/s), default: 0
%
%   Outputs:
%       g_vector    - 3x1 effective gravity vector in sample frame (m/s^2)
%       g_magnitude - Scalar magnitude in g-units (1g = 9.80665 m/s^2)
%
%   Theory:
%       The effective gravity experienced by the sample includes:
%       1. Gravitational acceleration transformed to rotating frame
%       2. Centrifugal acceleration: -omega x (omega x r)
%
%       For microgravity simulation, we want the time-averaged magnitude
%       of g_vector to approach zero.
%
%   Example:
%       % Simple case - gravity only
%       [gv, gm] = compute_gravity_vector(pi/4, pi/6);
%       fprintf('Effective g: %.4f g\n', gm);
%
%       % With centrifugal effects
%       pos = [0; 0; 0.1];
%       [gv, gm] = compute_gravity_vector(pi/4, pi/6, pos, 2*pi/60, 2*pi/60);
%
%   See also: COMPUTE_ROTATION_MATRIX, SIMULATE_RPM

%% Constants
g0 = 9.80665;  % Standard gravity (m/s^2)
g_lab = [0; -g0; 0];  % Gravity vector in lab frame (pointing down)

%% Handle optional inputs
if nargin < 3 || isempty(position)
    position = [0; 0; 0.1];  % Default sample position
end

if nargin < 4
    omega_inner = 0;
end

if nargin < 5
    omega_outer = 0;
end

%% Validate inputs
arguments
    theta_inner (1,1) double {mustBeReal}
    theta_outer (1,1) double {mustBeReal}
    position (3,1) double {mustBeReal} = [0; 0; 0.1]
    omega_inner (1,1) double {mustBeReal} = 0
    omega_outer (1,1) double {mustBeReal} = 0
end

%% Compute rotation matrix
R = compute_rotation_matrix(theta_inner, theta_outer);

%% Transform gravity to sample frame
% The gravity vector as seen from the rotating sample frame
% g_sample = R' * g_lab (transpose because we go from lab to sample)
g_gravity = R' * g_lab;

%% Compute centrifugal acceleration if angular velocities are non-zero
if abs(omega_inner) > 1e-10 || abs(omega_outer) > 1e-10
    % Angular velocity vectors in lab frame
    omega_inner_lab = [omega_inner; 0; 0];  % Inner rotates around X
    
    % Transform inner omega through outer frame rotation
    Ry = compute_rotation_matrix(0, theta_outer);  % Just outer rotation
    omega_inner_transformed = Ry * omega_inner_lab;
    
    % Outer frame omega
    omega_outer_lab = [0; omega_outer; 0];  % Outer rotates around Y
    
    % Total angular velocity in lab frame
    omega_total = omega_inner_transformed + omega_outer_lab;
    
    % Position in lab frame
    position_lab = R * position;
    
    % Centrifugal acceleration: a_cent = -omega x (omega x r)
    % This is the acceleration experienced in the rotating frame
    a_cent_lab = -cross(omega_total, cross(omega_total, position_lab));
    
    % Transform to sample frame
    a_centrifugal = R' * a_cent_lab;
else
    a_centrifugal = [0; 0; 0];
end

%% Total effective gravity
g_vector = g_gravity + a_centrifugal;

%% Compute magnitude in g-units
g_magnitude = norm(g_vector) / g0;

end
