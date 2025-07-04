# 粒子群算法(PSO)优化PID参数
import openpyxl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso
from scipy import signal
from scipy.optimize import differential_evolution, minimize
from sklearn.metrics import mean_squared_error

data = pd.read_excel('D:/B任务数据集.xlsx')
time = data["time"].values
temperature = data["temperature"].values
voltage = data["volte"].values
# 参数辨识
y0 = temperature[0]
y_ss = temperature[21600]  # 修正后的稳态值
delta_u = 3.5  # 阶跃输入幅度
K = (y_ss - y0) / delta_u  # 静态增益

# 找到28.3%和63.2%响应点的时间
y_28 = y0 + 0.283 * (y_ss - y0)
y_63 = y0 + 0.632 * (y_ss - y0)

# 插值找到对应时间
t_28 = np.interp(y_28, temperature, time)
t_63 = np.interp(y_63, temperature, time)
tau =1.5*( t_63 - t_28 ) # 时间常数
theta = 0  # 根据一阶模型，延迟时间为0
Kp_zn = 1.2 * tau / (K * theta) if theta != 0 else 0.6 * tau / K  # 修正公式，当θ=0时
Ti_zn = 2 * theta if theta != 0 else tau  # 修正公式
Td_zn = 0.5 * theta if theta != 0 else 0.25 * tau  # 修正公式
def pid_controller(Kp, Ki, Kd, setpoint, process_var, prev_error, integral, prev_measurement):
    error = setpoint - process_var
    integral += error
    derivative = (process_var - prev_measurement)
    output = Kp * error + Ki * integral + Kd * derivative
    return output, error, integral

def simulate_pid(Kp, Ki, Kd, setpoint=35):
    # 初始化变量
    output = np.zeros_like(time)
    process_var = np.zeros_like(time)
    process_var[0] = temperature[0]
    integral = 0
    prev_error = 0
    
    for i in range(1, len(time)):
        # PID计算
        control_signal, prev_error, integral = pid_controller(Kp, Ki, Kd, setpoint, process_var[i-1], prev_error, integral, process_var[i-1])
        
        # 限制控制信号范围 (假设电压范围0-10V)
        control_signal = np.clip(control_signal, 0, 10)
        
        # 系统响应 (使用辨识得到的一阶模型)
        dt = time[i] - time[i-1]
        process_var[i] = process_var[i-1] + (K * control_signal - process_var[i-1]) / tau * dt
    
    return process_var

# 定义目标函数（与之前相同）
def objective_function(params):
    Kp, Ki, Kd = params
    response = simulate_pid(Kp, Ki, Kd)
    
    # 计算性能指标
    steady_state_error = np.abs(response[-100:] - 35).mean()  # 稳态误差
    overshoot = np.max(response) - 35 if np.max(response) > 35 else 0  # 超调量
    settling_time_idx = np.where(np.abs(response - 35) <= 0.02 * 35)[0]
    settling_time = time[settling_time_idx[0]] if len(settling_time_idx) > 0 else time[-1]
    
    # 综合目标函数
    cost = 0.5 * steady_state_error + 0.3 * overshoot + 0.2 * (settling_time / time[-1])
    return cost

# 定义参数范围（与DE相同）
lb = [0.1 * Kp_zn, 0.1 * (Kp_zn/Ti_zn), 0.1 * (Kp_zn*Td_zn)]  # 下限
ub = [10 * Kp_zn, 10 * (Kp_zn/Ti_zn), 10 * (Kp_zn*Td_zn)]     # 上限

# PSO优化
print("\nRunning PSO optimization...")
optimized_params_pso, min_cost = pso(objective_function, lb, ub,swarmsize=30, maxiter=50,omega=0.5, phip=0.5, phig=0.5,debug=True)

print(f"\nOptimized parameters (PSO):")
print(f"Kp = {optimized_params_pso[0]:.4f}")
print(f"Ki = {optimized_params_pso[1]:.4f}")
print(f"Kd = {optimized_params_pso[2]:.4f}")
print(f"Minimum cost: {min_cost:.4f}")

# 使用PSO优化参数仿真
response_pso = simulate_pid(*optimized_params_pso)

# 绘制响应曲线
plt.figure(figsize=(12, 6))
plt.plot(time, response_pso, label='PID Response (PSO Optimized)')
plt.axhline(y=35, color='r', linestyle='--', label='Setpoint (35°C)')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Closed-loop Response with PSO Optimized PID')
plt.legend()
plt.grid(True)
plt.show()

# 更新性能比较表格
metrics_pso = pso.calculate_performance_metrics(response_pso)

# 添加到之前的比较表格
metrics_df = pd.DataFrame({'Ziegler-Nichols': pso.metrics_zn,'Differential Evolution': pso.metrics_de,'Simulated Annealing': pso.metrics_sa,'Particle Swarm': metrics_pso}).T

print("\nPerformance Metrics Comparison (Including PSO):")
print(metrics_df)

# 绘制所有方法响应比较
plt.figure(figsize=(12, 6))
plt.plot(time, pso.response_zn, '--', label='Ziegler-Nichols PID', alpha=0.7)
plt.plot(time, pso.response_de, '-.', label='DE Optimized PID', alpha=0.7)
plt.plot(time, pso.response_sa, ':', label='SA Optimized PID', alpha=0.7)
plt.plot(time, response_pso, '-', label='PSO Optimized PID', linewidth=2)
plt.axhline(y=35, color='r', linestyle='--', label='Setpoint (35°C)')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Comparison of All PID Controller Responses')
plt.legend()
plt.grid(True)
plt.show()