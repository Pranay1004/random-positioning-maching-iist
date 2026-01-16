function R = compute_rotation_matrix(theta_inner, theta_outer)
%COMPUTE_ROTATION_MATRIX Compute combined rotation matrix for RPM
%
%   R = COMPUTE_ROTATION_MATRIX(theta_inner, theta_outer) computes the
%   combined rotation matrix for the Random Positioning Machine.
%
%   The RPM has two rotation axes:
%   - Inner frame: Rotates around X-axis
%   - Outer frame: Rotates around Y-axis
%
%   The combined rotation is: R = Ry(theta_outer) * Rx(theta_inner)
%
%   Inputs:
%       theta_inner - Inner frame angle (radians)
%       theta_outer - Outer frame angle (radians)
%
%   Output:
%       R - 3x3 rotation matrix
%
%   Example:
%       R = compute_rotation_matrix(pi/4, pi/6);
%       g_sample = R' * [0; -9.81; 0];  % Gravity in sample frame
%
%   See also: COMPUTE_GRAVITY_VECTOR, SIMULATE_RPM

%% Validate inputs
arguments
    theta_inner (1,1) double {mustBeReal}
    theta_outer (1,1) double {mustBeReal}
end

%% Compute individual rotation matrices

% Rotation around X-axis (inner frame)
c_i = cos(theta_inner);
s_i = sin(theta_inner);
Rx = [1,   0,    0;
      0, c_i, -s_i;
      0, s_i,  c_i];

% Rotation around Y-axis (outer frame)
c_o = cos(theta_outer);
s_o = sin(theta_outer);
Ry = [c_o,  0, s_o;
       0,   1,   0;
     -s_o,  0, c_o];

%% Combined rotation
% Order: First rotate inner frame, then outer frame
R = Ry * Rx;

end
