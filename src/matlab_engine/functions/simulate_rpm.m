function results = simulate_rpm(duration, dt, omega_inner, omega_outer, options)
%SIMULATE_RPM Simulate RPM operation and compute microgravity metrics
%
%   results = SIMULATE_RPM(duration, dt, omega_inner, omega_outer)
%   simulates the Random Positioning Machine for the specified duration
%   and computes gravity metrics.
%
%   results = SIMULATE_RPM(duration, dt, omega_inner, omega_outer, options)
%   uses additional options specified as name-value pairs.
%
%   Inputs:
%       duration    - Simulation duration (seconds)
%       dt          - Time step (seconds)
%       omega_inner - Inner frame angular velocity (rad/s or RPM, see options)
%       omega_outer - Outer frame angular velocity (rad/s or RPM, see options)
%
%   Options:
%       'Unit'          - 'rad/s' or 'rpm' (default: 'rad/s')
%       'Position'      - Sample position [x;y;z] in meters (default: [0;0;0.1])
%       'Mode'          - 'constant', 'random' (default: 'constant')
%       'PlotResults'   - true/false (default: true)
%       'RandomSeed'    - Seed for random mode (default: random)
%
%   Output:
%       results - Structure containing:
%           .time           - Time vector
%           .theta_inner    - Inner frame angles
%           .theta_outer    - Outer frame angles
%           .omega_inner    - Inner frame velocities
%           .omega_outer    - Outer frame velocities
%           .g_magnitude    - Instantaneous g values
%           .g_vector       - Gravity vectors (3 x N)
%           .mean_g         - Time-averaged g magnitude
%           .std_g          - Standard deviation of g
%           .max_g          - Maximum g
%           .quality        - Quality assessment string
%
%   Example:
%       % 3D clinostat mode at 2 RPM
%       results = simulate_rpm(60, 0.01, 2, 2, 'Unit', 'rpm');
%       fprintf('Mean g: %.4f\n', results.mean_g);
%
%       % Random mode
%       results = simulate_rpm(120, 0.01, 5, 5, 'Unit', 'rpm', 'Mode', 'random');
%
%   See also: COMPUTE_GRAVITY_VECTOR, ANALYZE_MICROGRAVITY

%% Parse inputs
arguments
    duration (1,1) double {mustBePositive}
    dt (1,1) double {mustBePositive}
    omega_inner (1,1) double
    omega_outer (1,1) double
    options.Unit (1,:) char {mustBeMember(options.Unit, {'rad/s', 'rpm'})} = 'rad/s'
    options.Position (3,1) double = [0; 0; 0.1]
    options.Mode (1,:) char {mustBeMember(options.Mode, {'constant', 'random'})} = 'constant'
    options.PlotResults (1,1) logical = true
    options.RandomSeed double = []
end

%% Convert units if needed
if strcmpi(options.Unit, 'rpm')
    omega_inner = omega_inner * 2 * pi / 60;  % Convert to rad/s
    omega_outer = omega_outer * 2 * pi / 60;
end

%% Initialize random number generator
if ~isempty(options.RandomSeed)
    rng(options.RandomSeed);
end

%% Initialize simulation
N = round(duration / dt);
time = (0:N-1)' * dt;

% Pre-allocate arrays
theta_inner = zeros(N, 1);
theta_outer = zeros(N, 1);
omega_i = zeros(N, 1);
omega_o = zeros(N, 1);
g_magnitude = zeros(N, 1);
g_vector = zeros(3, N);

% Initial conditions
theta_inner(1) = 0;
theta_outer(1) = 0;
omega_i(1) = omega_inner;
omega_o(1) = omega_outer;

% Random mode parameters
if strcmpi(options.Mode, 'random')
    next_change = 0;
    omega_range = [0.5, 5] * 2 * pi / 60;  % 0.5-5 RPM in rad/s
    change_interval = [2, 10];  % seconds
end

%% Main simulation loop
fprintf('Simulating RPM for %.1f seconds...\n', duration);
tic;

for i = 1:N
    t = time(i);
    
    % Update velocities based on mode
    if strcmpi(options.Mode, 'random') && t >= next_change
        % Random direction and speed change
        omega_i(i) = (rand * diff(omega_range) + omega_range(1)) * sign(randn);
        omega_o(i) = (rand * diff(omega_range) + omega_range(1)) * sign(randn);
        next_change = t + rand * diff(change_interval) + change_interval(1);
    else
        if i > 1
            omega_i(i) = omega_i(i-1);
            omega_o(i) = omega_o(i-1);
        end
    end
    
    % Update positions
    if i > 1
        theta_inner(i) = theta_inner(i-1) + omega_i(i) * dt;
        theta_outer(i) = theta_outer(i-1) + omega_o(i) * dt;
        
        % Wrap to [-pi, pi]
        theta_inner(i) = atan2(sin(theta_inner(i)), cos(theta_inner(i)));
        theta_outer(i) = atan2(sin(theta_outer(i)), cos(theta_outer(i)));
    end
    
    % Compute gravity
    [gv, gm] = compute_gravity_vector(theta_inner(i), theta_outer(i), ...
        options.Position, omega_i(i), omega_o(i));
    g_vector(:, i) = gv;
    g_magnitude(i) = gm;
end

elapsed = toc;
fprintf('Simulation completed in %.2f seconds (%.1fx real-time)\n', elapsed, duration/elapsed);

%% Compute statistics
mean_g = mean(g_magnitude);
std_g = std(g_magnitude);
max_g = max(g_magnitude);
min_g = min(g_magnitude);

% Quality assessment
if mean_g < 0.01
    quality = 'Excellent';
elseif mean_g < 0.05
    quality = 'Good';
elseif mean_g < 0.1
    quality = 'Acceptable';
else
    quality = 'Poor';
end

%% Store results
results = struct();
results.time = time;
results.theta_inner = theta_inner;
results.theta_outer = theta_outer;
results.omega_inner = omega_i;
results.omega_outer = omega_o;
results.g_magnitude = g_magnitude;
results.g_vector = g_vector;
results.mean_g = mean_g;
results.std_g = std_g;
results.max_g = max_g;
results.min_g = min_g;
results.quality = quality;
results.position = options.Position;
results.mode = options.Mode;

%% Display summary
fprintf('\n=== Simulation Results ===\n');
fprintf('Duration: %.1f s\n', duration);
fprintf('Samples: %d\n', N);
fprintf('Mode: %s\n', options.Mode);
fprintf('Position: [%.3f, %.3f, %.3f] m\n', options.Position);
fprintf('\nGravity Statistics:\n');
fprintf('  Mean g: %.5f g\n', mean_g);
fprintf('  Std g:  %.5f g\n', std_g);
fprintf('  Min g:  %.5f g\n', min_g);
fprintf('  Max g:  %.5f g\n', max_g);
fprintf('  Quality: %s\n', quality);

%% Plot results
if options.PlotResults
    plot_rpm_results(results);
end

end

function plot_rpm_results(results)
%PLOT_RPM_RESULTS Create visualization of simulation results

figure('Position', [100, 100, 1400, 900], 'Color', [0.1 0.1 0.15]);

% Gravity magnitude over time
subplot(2, 3, 1);
plot(results.time, results.g_magnitude, 'Color', [0 1 0.5], 'LineWidth', 1);
hold on;
yline(results.mean_g, '--', 'Color', [1 0.5 0], 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Effective g (g-units)');
title('Instantaneous Gravity');
legend('g(t)', sprintf('Mean = %.4f g', results.mean_g), 'Location', 'best');
grid on;
set(gca, 'Color', [0.05 0.05 0.1], 'XColor', 'w', 'YColor', 'w');

% Frame angles
subplot(2, 3, 2);
plot(results.time, rad2deg(results.theta_inner), 'Color', [0 0.8 1], 'LineWidth', 1);
hold on;
plot(results.time, rad2deg(results.theta_outer), 'Color', [1 0.4 0.4], 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Angle (degrees)');
title('Frame Positions');
legend('\theta_{inner}', '\theta_{outer}', 'Location', 'best');
grid on;
set(gca, 'Color', [0.05 0.05 0.1], 'XColor', 'w', 'YColor', 'w');

% Angular velocities
subplot(2, 3, 3);
plot(results.time, results.omega_inner * 60 / (2*pi), 'Color', [0 0.8 1], 'LineWidth', 1);
hold on;
plot(results.time, results.omega_outer * 60 / (2*pi), 'Color', [1 0.4 0.4], 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Velocity (RPM)');
title('Angular Velocities');
legend('\omega_{inner}', '\omega_{outer}', 'Location', 'best');
grid on;
set(gca, 'Color', [0.05 0.05 0.1], 'XColor', 'w', 'YColor', 'w');

% Gravity histogram
subplot(2, 3, 4);
histogram(results.g_magnitude, 50, 'FaceColor', [0 1 0.5], 'EdgeColor', 'none');
xlabel('Effective g (g-units)');
ylabel('Count');
title('Gravity Distribution');
grid on;
set(gca, 'Color', [0.05 0.05 0.1], 'XColor', 'w', 'YColor', 'w');

% 3D gravity vector path
subplot(2, 3, 5);
% Normalize for visualization
gv_norm = results.g_vector / 9.80665;  % Convert to g-units
plot3(gv_norm(1,:), gv_norm(2,:), gv_norm(3,:), 'Color', [0 1 0.5], 'LineWidth', 0.5);
hold on;
% Plot unit sphere
[X, Y, Z] = sphere(20);
surf(X, Y, Z, 'FaceAlpha', 0.1, 'EdgeColor', [0.3 0.3 0.3], 'FaceColor', [0.2 0.2 0.3]);
xlabel('g_x (g)');
ylabel('g_y (g)');
zlabel('g_z (g)');
title('Gravity Vector Path');
axis equal;
grid on;
set(gca, 'Color', [0.05 0.05 0.1], 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w');
view(30, 20);

% Summary text
subplot(2, 3, 6);
axis off;
text(0.1, 0.9, 'SIMULATION SUMMARY', 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'w');
text(0.1, 0.75, sprintf('Mode: %s', results.mode), 'FontSize', 11, 'Color', 'w');
text(0.1, 0.65, sprintf('Duration: %.1f s', results.time(end)), 'FontSize', 11, 'Color', 'w');
text(0.1, 0.55, sprintf('Sample Position: [%.2f, %.2f, %.2f] m', results.position), 'FontSize', 11, 'Color', 'w');
text(0.1, 0.40, 'Microgravity Metrics:', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'w');
text(0.1, 0.30, sprintf('  Mean g: %.5f g', results.mean_g), 'FontSize', 11, 'Color', [0 1 0.5]);
text(0.1, 0.20, sprintf('  Std g: %.5f g', results.std_g), 'FontSize', 11, 'Color', 'w');
text(0.1, 0.10, sprintf('  Max g: %.5f g', results.max_g), 'FontSize', 11, 'Color', 'w');

% Quality badge
if strcmp(results.quality, 'Excellent')
    badge_color = [0 0.8 0.3];
elseif strcmp(results.quality, 'Good')
    badge_color = [0.2 0.6 1];
elseif strcmp(results.quality, 'Acceptable')
    badge_color = [1 0.7 0];
else
    badge_color = [1 0.3 0.3];
end
rectangle('Position', [0.5, 0.6, 0.4, 0.15], 'Curvature', 0.3, ...
    'FaceColor', badge_color, 'EdgeColor', 'none');
text(0.7, 0.675, results.quality, 'FontSize', 14, 'FontWeight', 'bold', ...
    'Color', 'w', 'HorizontalAlignment', 'center');

set(gca, 'Color', [0.1 0.1 0.15]);

sgtitle('RPM Digital Twin - Simulation Results', 'Color', 'w', 'FontSize', 16);

end
