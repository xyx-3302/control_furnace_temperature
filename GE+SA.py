# 加载数据
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

# 一阶模型阶跃响应
def model_step_response(t, K, tau, y0, delta_u):
    return y0 + K * delta_u * (1 - np.exp(-t / tau))

# 生成模型预测值
t_model = np.linspace(0, time[-1], 1000)
y_model = model_step_response(t_model, K, tau, y0, delta_u)

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(time, temperature, label="Actual Data", linewidth=2)
plt.plot(t_model, y_model, label=f"Model: K={K:.2f} °C/V, τ={tau:.2f} s", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Step Response of Heating System (Model vs Actual)")
plt.legend()
plt.grid(True)
plt.show()

# 输出参数
print(f"Identified Parameters:")
print(f"- Static Gain K = {K:.2f} °C/V")
print(f"- Time Constant τ = {tau:.2f} s")
# 计算模型与实际响应之间的均方误差(MSE)
actual_response = temperature
model_response = model_step_response(time, K, tau, y0, delta_u)

mse = mean_squared_error(actual_response, model_response)
print(f"Model MSE: {mse:.4f}")

# 3. PID控制器设计
# 3.1 初始PID参数设计 (Ziegler-Nichols方法)
theta = 0  # 根据一阶模型，延迟时间为0
Kp_zn = 1.2 * tau / (K * theta) if theta != 0 else 0.6 * tau / K  # 修正公式，当θ=0时
Ti_zn = 2 * theta if theta != 0 else tau  # 修正公式
Td_zn = 0.5 * theta if theta != 0 else 0.25 * tau  # 修正公式

print(f"\nZiegler-Nichols parameters:")
print(f"Kp = {Kp_zn:.4f}")
print(f"Ti = {Ti_zn:.4f} s")
print(f"Td = {Td_zn:.4f} s")
print(f"Ki = {Kp_zn/Ti_zn:.4f} (1/s)")
print(f"Kd = {Kp_zn*Td_zn:.4f} s")

# 3.2 闭环系统仿真
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
        control_signal, prev_error, integral = pid_controller(
            Kp, Ki, Kd, setpoint, process_var[i-1], prev_error, integral, process_var[i-1])
        
        # 限制控制信号范围 (假设电压范围0-10V)
        control_signal = np.clip(control_signal, 0, 10)
        
        # 系统响应 (使用辨识得到的一阶模型)
        dt = time[i] - time[i-1]
        process_var[i] = process_var[i-1] + (K * control_signal - process_var[i-1]) / tau * dt
    
    return process_var

# 使用Ziegler-Nichols参数进行仿真
Kp = Kp_zn
Ki = Kp_zn / Ti_zn
Kd = Kp_zn * Td_zn

response_zn = simulate_pid(Kp, Ki, Kd)

plt.figure(figsize=(12, 6))
plt.plot(time, response_zn, label='PID Response (Ziegler-Nichols)')
plt.plot(time, temperature, '--', alpha=0.5, label='Open-loop Response')
plt.axhline(y=35, color='r', linestyle='--', label='Setpoint (35°C)')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Closed-loop Response with Ziegler-Nichols PID')
plt.legend()
plt.grid(True)
plt.show()
#差分进化优化
def objective_function(params):
    Kp, Ki, Kd = params
    response = simulate_pid(Kp, Ki, Kd)
    
    # 计算性能指标
    steady_state_error = np.abs(response[-100:] - 35).mean()  # 稳态误差
    overshoot = np.max(response) - 35 if np.max(response) > 35 else 0  # 超调量
    settling_time_idx = np.where(np.abs(response - 35) <= 0.02 * 35)[0]
    settling_time = time[settling_time_idx[0]] if len(settling_time_idx) > 0 else time[-1]
    
    # 综合目标函数 (权重可根据需求调整)
    cost = 0.5 * steady_state_error + 0.3 * overshoot + 0.2 * (settling_time / time[-1])
    return cost

# 定义参数范围 (基于Ziegler-Nichols参数扩展)
bounds = [(0.1 * Kp_zn, 10 * Kp_zn), 
          (0.1 * (Kp_zn/Ti_zn), 10 * (Kp_zn/Ti_zn)), 
          (0.1 * (Kp_zn*Td_zn), 10 * (Kp_zn*Td_zn))]

# 差分进化优化
result_de = differential_evolution(objective_function, bounds,strategy='best1bin', maxiter=50, popsize=15, tol=1e-3, seed=42)
optimized_params_de = result_de.x
print(f"\nOptimized parameters (DE):")
print(f"Kp = {optimized_params_de[0]:.4f}")
print(f"Ki = {optimized_params_de[1]:.4f}")
print(f"Kd = {optimized_params_de[2]:.4f}")

# 使用优化参数仿真
response_de = simulate_pid(*optimized_params_de)

plt.figure(figsize=(12, 6))
plt.plot(time, response_de, label='PID Response (DE Optimized)')
plt.axhline(y=35, color='r', linestyle='--', label='Setpoint (35°C)')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Closed-loop Response with DE Optimized PID')
plt.legend()
plt.grid(True)
plt.show()
# 模拟退火优化 (使用Nelder-Mead方法作为替代)
initial_params = [Kp_zn, Kp_zn/Ti_zn, Kp_zn*Td_zn]  # 从Ziegler-Nichols参数开始

result_sa = minimize(objective_function, initial_params, method='Nelder-Mead',
                    bounds=bounds, options={'maxiter': 100, 'xatol': 1e-3})
optimized_params_sa = result_sa.x
print(f"\nOptimized parameters (SA):")
print(f"Kp = {optimized_params_sa[0]:.4f}")
print(f"Ki = {optimized_params_sa[1]:.4f}")
print(f"Kd = {optimized_params_sa[2]:.4f}")

# 使用优化参数仿真
response_sa = simulate_pid(*optimized_params_sa)

plt.figure(figsize=(12, 6))
plt.plot(time, response_sa, label='PID Response (SA Optimized)')
plt.axhline(y=35, color='r', linestyle='--', label='Setpoint (35°C)')
plt.xlabel('Time (s)')
plt.ylabel('Temperature (°C)')
plt.title('Closed-loop Response with SA Optimized PID')
plt.legend()
plt.grid(True)
plt.show()
print("\nRunning PSO optimization...")
lb = [0.1 * Kp_zn, 0.1 * (Kp_zn/Ti_zn), 0.1 * (Kp_zn*Td_zn)]  # 下限
ub = [10 * Kp_zn, 10 * (Kp_zn/Ti_zn), 10 * (Kp_zn*Td_zn)]     # 上限
optimized_params_pso, min_cost = pso(objective_function, lb, ub,swarmsize=30, maxiter=50,omega=0.5, phip=0.5, phig=0.5,debug=True)

print(f"\nOptimized parameters (PSO):")
print(f"Kp = {optimized_params_pso[0]:.4f}")
print(f"Ki = {optimized_params_pso[1]:.4f}")
print(f"Kd = {optimized_params_pso[2]:.4f}")
print(f"Minimum cost: {min_cost:.4f}")
