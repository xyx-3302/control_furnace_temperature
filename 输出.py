import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from collections import deque

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, window_size=100):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0
        self.prev_error = 0
        self.prev_time = 0
        self.error_window = deque(maxlen=window_size)
        self.time_window = deque(maxlen=window_size)

    def compute(self, current_value, current_time):
        error = self.setpoint - current_value
        self.error_window.append(error)
        self.time_window.append(current_time)

        # Proportional term
        P = self.Kp * error

        # Integral term with anti-windup
        if len(self.time_window) > 1:
            dt = current_time - self.prev_time
            self.integral += error * dt
            max_integral = 100 / (self.Ki + 1e-6)
            self.integral = np.clip(self.integral, -max_integral, max_integral)

        I = self.Ki * self.integral

        # Derivative term
        if len(self.time_window) > 1 and (current_time - self.prev_time) > 1e-6:
            de = error - self.prev_error
            dt = current_time - self.prev_time
            D = self.Kd * de / dt
        else:
            D = 0

        self.prev_error = error
        self.prev_time = current_time

        control_output = P + I + D
        return np.clip(control_output, -10, 10)

def system_dynamics(t, y, pid):
    u = pid.compute(y, t)
    dydt = (K * u - y + y0) / tau
    return dydt

def simulate_system(pid_params, simulation_time=1000):
    Kp, Ki, Kd = pid_params
    pid = PIDController(Kp, Ki, Kd, setpoint=35)

    sol = solve_ivp(system_dynamics, [0, simulation_time], [y0],
                    args=(pid,), method='BDF',
                    dense_output=True, rtol=1e-6, atol=1e-8)

    t_eval = np.linspace(0, simulation_time, min(simulation_time, 1000))
    y = sol.sol(t_eval)[0]

    return t_eval, y

def calculate_performance(t, y, setpoint=35):
    # Steady-state analysis (last 20% of simulation)
    steady_state_start = int(0.8 * len(t))
    y_ss = np.mean(y[steady_state_start:])
    e_ss = np.abs(setpoint - y_ss)

    # Dynamic performance metrics
    y_final = y[-1]
    y_range = y_final - y[0]

    # Rise time (10% to 90%)
    idx_10 = np.argmax(y >= y[0] + 0.1 * y_range)
    idx_90 = np.argmax(y >= y[0] + 0.9 * y_range)
    t_r = t[idx_90] - t[idx_10] if idx_90 > idx_10 else np.nan

    # Peak overshoot
    y_max = np.max(y)
    M_p = (y_max - y_final) / (y_final - y[0]) * 100 if y_final != y[0] else 0

    # Settling time (within ±5% of final value)
    settling_bound = 0.05 * (y_final - y[0])
    within_bounds = np.abs(y - y_final) <= settling_bound
    crossings = np.where(np.diff(within_bounds.astype(int)) != 0)[0]
    t_s = t[crossings[-1]] if len(crossings) > 0 else t[-1]

    # ITAE performance index
    e = setpoint - y
    itae = np.trapz(t * np.abs(e), t)

    return {
        'steady_state': y_ss,
        'steady_state_error': e_ss,
        'rise_time': t_r,
        'overshoot': M_p,
        'settling_time': t_s,
        'itae': itae,
        'response': y
    }

# 系统参数
K = 9.85  # 系统增益
tau = 2006.34  # 时间常数
y0 = 16.85  # 初始温度

# 创建图形
plt.figure(figsize=(14, 8))

# 1. DE优化结果 (示例参数)
DE_params = [1832.7165,0.0061,992702.7580]
t_DE, y_DE = simulate_system(DE_params, 1500)
perf_DE = calculate_performance(t_DE, y_DE)
plt.plot(t_DE, y_DE, 'b-', linewidth=2, label=f'DE Optimized (Kp={DE_params[0]:.2f}, Ki={DE_params[1]:.4f}, Kd={DE_params[2]:.2f})')
# 2. SA调节结果 (示例参数)
SA_params = [1832.8429, 0.0061,855740.2062]
t_SA, y_SA = simulate_system(SA_params, 1500)
perf_SA = calculate_performance(t_SA, y_SA)
plt.plot(t_SA, y_SA, 'g--', linewidth=2, label=f'SA Tuning (Kp={SA_params[0]:.2f}, Ki={SA_params[1]:.4f}, Kd={SA_params[2]:.2f})')
# 3. pso优化结果 (示例参数)
pso_params = [1832.8429, 0.0061,301087.7328]
t_pso, y_pso = simulate_system(pso_params, 1500)
perf_pso = calculate_performance(t_pso, y_pso)
plt.plot(t_pso, y_pso, 'r-.', linewidth=2, label=f'pso Optimized (Kp={pso_params[0]:.2f}, Ki={pso_params[1]:.4f}, Kd={pso_params[2]:.2f})')
# 4. common优化结果 (示例参数)
common_params = [2,0.01,1000]
t_common, y_common = simulate_system(common_params, 1500)
perf_common = calculate_performance(t_common, y_common)
plt.plot(t_common, y_common, 'b-', linewidth=2, label=f'common Optimized (Kp={common_params[0]:.2f}, Ki={common_params[1]:.4f}, Kd={common_params[2]:.2f})')
# Add reference lines and decorations
plt.axhline(35, color='k', linestyle=':', label='Setpoint (35°C)')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.title('Comparison of Three PID Control Methods', fontsize=14)
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# 2. Create performance metrics table separately
plt.figure(figsize=(10, 4))
performance_data = [
    ["", "DE Optimized", "SAl Tuning", "pso Optimized","common Optimized"],
    ["Steady-state Error (°C)", f"{perf_DE['steady_state_error']:.4f}", f"{perf_SA['steady_state_error']:.4f}", f"{perf_pso['steady_state_error']:.4f}", f"{perf_common['steady_state_error']:.4f}"],
    ["Rise Time (s)", f"{perf_DE['rise_time']:.2f}", f"{perf_SA['rise_time']:.2f}", f"{perf_pso['rise_time']:.2f}",f"{perf_common['rise_time']:.2f}"],
    ["Overshoot (%)", f"{perf_DE['overshoot']:.2f}", f"{perf_SA['overshoot']:.2f}", f"{perf_pso['overshoot']:.2f}", f"{perf_common['overshoot']:.2f}"],
    ["Settling Time (s)", f"{perf_DE['settling_time']:.2f}", f"{perf_SA['settling_time']:.2f}", f"{perf_pso['settling_time']:.2f}", f"{perf_common['settling_time']:.2f}"],
    ["ITAE Index", f"{perf_DE['itae']:.2f}", f"{perf_SA['itae']:.2f}", f"{perf_pso['itae']:.2f}", f"{perf_common['itae']:.2f}"]
]

# Create table
table = plt.table(cellText=performance_data,
                 colWidths=[0.2, 0.2, 0.2, 0.2,0.2],
                 loc='center',
                 cellLoc='center')

# Adjust table style
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)  # Adjust row height

# Hide axes
plt.axis('off')
plt.title('Performance Metrics Comparison', fontsize=14, pad=20)

plt.tight_layout()
plt.show()