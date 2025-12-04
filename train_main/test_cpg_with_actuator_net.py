import os
import sys
import numpy as np
import math
import random
import torch

# --- 1. 自動設定 Webots 路徑 ---
if 'WEBOTS_HOME' not in os.environ:
    default_path = r"C:\Program Files\Webots"
    if os.path.exists(default_path):
        os.environ['WEBOTS_HOME'] = default_path
    else:
        print("錯誤: 找不到 WEBOTS_HOME")
        sys.exit(1)

webots_python_path = os.path.join(os.environ['WEBOTS_HOME'], 'lib', 'controller', 'python')
if webots_python_path not in sys.path:
    sys.path.append(webots_python_path)

from controller import Supervisor 
from actuator_net import ActuatorModel

# --- 2. 初始化 ---
robot = Supervisor() 
timestep = int(robot.getBasicTimeStep())
self_node = robot.getSelf()
if self_node is None: self_node = robot.getFromDef("MyPlenBot")

try:
    print("正在載入 ActuatorNet 模型...")
    actuator = ActuatorModel('mg90s_actuator_net_optimized.pth')
except Exception as e:
    print(f"錯誤: {e}")
    sys.exit(1)

motor_names = [
    "lb_servo_l_hip", "l_hip_l_thigh", "l_thigh_l_knee", "l_knee_l_shin", "l_shin_l_ankle", "l_ankle_l_foot",
    "rb_servo_r_hip", "r_hip_r_thigh", "r_thigh_r_knee", "r_knee_r_shin", "r_shin_r_ankle", "r_ankle_r_foot",
    "torso_l_shoulder", "l_shoulder_ls_servo", "le_servo_l_elbow",
    "torso_r_shoulder", "r_shoulder_rs_servo", "re_servo_r_elbow",
]

motors = []
motor_limits = []
num_motors = len(motor_names)
SERVO_OFFSET = 70.0 
SAFETY_MARGIN = 1e-3

print("\n--- 馬達初始化 ---")
for name in motor_names:
    m = robot.getDevice(name)
    motors.append(m)
    if m:
        min_p = m.getMinPosition()
        max_p = m.getMaxPosition()
        if min_p == 0 and max_p == 0: min_p, max_p = -1.57, 1.57
        motor_limits.append((min_p + SAFETY_MARGIN, max_p - SAFETY_MARGIN))
    else:
        motor_limits.append((-1.57, 1.57))

# --- 3. 變數與參數 ---
hist_real = np.full((num_motors, actuator.hist_len), SERVO_OFFSET)
hist_cmd = np.full((num_motors, actuator.cmd_len), SERVO_OFFSET)

# CPG 參數 (使用您測試成功的大振幅)
walk_freq = 2.0       
amp_hip_pitch = 0.6  
amp_knee_pitch = 0.6
amp_hip_roll = 0.3
amp_ankle_roll = 0.20
forward_lean_offset = 0.1

# [新增] 持續推力參數
PUSH_INTERVAL_SEC = 4.0   # 每 4 秒推一次 (給它時間恢復)
PUSH_DURATION_SEC = 0.4   # 每次推力持續 0.5 秒
FORCE_MAGNITUDE = 0.7         # 力的大小 (牛頓) - 持續推力要小一點，不然必倒

push_interval_steps = int(PUSH_INTERVAL_SEC * 1000 / timestep)
push_duration_steps = int(PUSH_DURATION_SEC * 1000 / timestep)

current_push_steps = 0    # 剩餘推力步數
current_force_vec = [0, 0, 0]

print(f"\n開始模擬: 每 {PUSH_INTERVAL_SEC} 秒施加持續 {PUSH_DURATION_SEC} 秒的推力...")
print("-" * 65)

step_count = 0

# --- 5. 主迴圈 ---
while robot.step(timestep) != -1:
    time = robot.getTime()
    step_count += 1
    
    # --- A. 推力產生邏輯 (觸發) ---
    if step_count % push_interval_steps == 0:
        # 隨機產生一個水平方向的力向量
        angle = random.uniform(0, 2 * math.pi)
        fx = FORCE_MAGNITUDE * math.cos(angle)
        fy = FORCE_MAGNITUDE * math.sin(angle)
        fz = 0
        
        current_force_vec = [fx, fy, fz]
        current_push_steps = push_duration_steps # 設定計時器
        print(f"{time:6.2f}s | \033[91m⚠️ START PUSH\033[0m ({PUSH_DURATION_SEC}s) | Vec: [{fx:.1f}, {fy:.1f}]")

    # --- B. 持續施力邏輯 (執行) ---
    if current_push_steps > 0:
        if self_node:
            # 每一幀都施加力，達成 "持續" 效果
            self_node.addForce(current_force_vec, False)
        current_push_steps -= 1
        
        # 顯示結束訊息
        if current_push_steps == 0:
             print(f"{time:6.2f}s | \033[92m✅ END PUSH\033[0m")

    # --- C. CPG 計算 ---
    phase_left = time * walk_freq - math.pi
    phase_right = time * walk_freq - math.pi
    phase_shift = time * walk_freq + math.pi / 2
    
    target_positions = np.zeros(num_motors)
    
    target_positions[2] = amp_hip_pitch * math.sin(phase_left) + forward_lean_offset 
    target_positions[8] = amp_hip_pitch * math.sin(phase_right) + forward_lean_offset
    target_positions[3] = amp_knee_pitch * max(0, math.sin(phase_left + math.pi/4))
    target_positions[9] = amp_knee_pitch * max(0, math.sin(phase_right + math.pi/4))
    
    roll = amp_hip_roll * math.sin(phase_shift)
    target_positions[1] = roll 
    target_positions[7] = roll 
    
    a_roll = amp_ankle_roll * math.sin(phase_shift)
    target_positions[5] = a_roll
    target_positions[11] = a_roll 
    
    target_positions[4] = 1 * (target_positions[2] + target_positions[3]) - forward_lean_offset
    target_positions[10] = 1 * (target_positions[8] + target_positions[9]) - forward_lean_offset
    
    # --- D. 安全夾擠 ---
    for i, (min_p, max_p) in enumerate(motor_limits):
        target_positions[i] = np.clip(target_positions[i], min_p, max_p)
        
    # --- E. ActuatorNet 推論 ---
    webots_deg = np.degrees(target_positions)
    servo_cmd_deg = webots_deg + SERVO_OFFSET
    
    hist_cmd = np.roll(hist_cmd, 1, axis=1)
    hist_cmd[:, 0] = servo_cmd_deg
    
    servo_real_deg = actuator.predict(hist_real, hist_cmd)
    
    hist_real = np.roll(hist_real, 1, axis=1)
    hist_real[:, 0] = servo_real_deg 
    
    webots_real_deg = servo_real_deg - SERVO_OFFSET
    webots_real_rad = np.radians(webots_real_deg)
    
    for i, (min_p, max_p) in enumerate(motor_limits):
        webots_real_rad[i] = np.clip(webots_real_rad[i], min_p, max_p)
        
    # --- F. 寫入 Webots ---
    for i, motor in enumerate(motors):
        if motor:
            motor.setPosition(webots_real_rad[i])