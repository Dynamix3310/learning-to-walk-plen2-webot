"""
Webots Python 控制器 (CPG 步行)
"""

from controller import Robot
import math

# --- 1. 初始化 ---
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# --- 2. 取得 .proto 檔案中定義的所有馬達裝置 ---
motors = {}
motor_names = [
    # 左腿 (6)
    "lb_servo_l_hip", "l_hip_l_thigh", "l_thigh_l_knee",
    "l_knee_l_shin", "l_shin_l_ankle", "l_ankle_l_foot",
    # 右腿 (6)
    "rb_servo_r_hip", "r_hip_r_thigh", "r_thigh_r_knee",
    "r_knee_r_shin", "r_shin_r_ankle", "r_ankle_r_foot",
    # 左臂 (3)
    "torso_l_shoulder", "l_shoulder_ls_servo", "le_servo_l_elbow",
    # 右臂 (3)
    "torso_r_shoulder", "r_shoulder_rs_servo", "re_servo_r_elbow",
]
for name in motor_names:
    motors[name] = robot.getDevice(name)

# --- 3. 設定 CPG (中央模式生成器) 參數 ---
walk_freq = 3.0       # 步行速度 (rad/s)

# 振幅 (Amplitude)
amp_hip_pitch = 0.3    # 大腿前後擺動
amp_knee_pitch = 0.3    # 膝蓋彎曲 (v7 修正)
amp_hip_roll = 0.1     # 臀部左右搖擺 (轉移重心)
amp_ankle_roll = 0.1    # 腳踝左右搖擺 (配合臀部)
amp_arm_swing = 0     # 手臂擺動
forward_lean_offset = -0.05

# --- 4. 主迴圈 ---
print("啟動 CPG 步行控制器 (v8 修正版)...")
while robot.step(timestep) != -1:
    # 獲取當前模擬時間
    time = robot.getTime()

    # --- 4.1 計算 CPG 訊號 (使用正弦波) ---
    phase_left_leg = time * walk_freq+math.pi
    phase_right_leg = time * walk_freq+math.pi
    phase_weight_shift = time * walk_freq + math.pi / 2

    # --- 4.2 計算每個關節的目標角度 ---

    # 1. 大腿前後 (Hip Pitch)
    l_hip_pitch = amp_hip_pitch * math.sin(phase_left_leg-math.pi) + forward_lean_offset
    r_hip_pitch = amp_hip_pitch * math.sin(phase_right_leg-math.pi) + forward_lean_offset
    motors["l_thigh_l_knee"].setPosition(l_hip_pitch)
    motors["r_thigh_r_knee"].setPosition(r_hip_pitch)

    # 2. 膝蓋彎曲 (Knee Pitch)
    l_knee = amp_knee_pitch * max(0, math.sin(phase_left_leg + math.pi / 4))
    r_knee = amp_knee_pitch * max(0, math.sin(phase_right_leg + math.pi / 4))
    motors["l_knee_l_shin"].setPosition(l_knee)
    motors["r_knee_r_shin"].setPosition(r_knee)

    # 3. 臀部左右 (Hip Roll) - 轉移重心
    hip_roll = amp_hip_roll * math.sin(phase_weight_shift)
    motors["l_hip_l_thigh"].setPosition(hip_roll)
    motors["r_hip_r_thigh"].setPosition(hip_roll)

    # 4. 腳踝左右 (Ankle Roll) - 輔助平衡
    ankle_roll = amp_ankle_roll * math.sin(phase_weight_shift)
    motors["l_ankle_l_foot"].setPosition(ankle_roll)
    motors["r_ankle_r_foot"].setPosition(ankle_roll) 

    # 5. 腳踝前後 (Ankle Pitch) - 輔助平衡
    # *** [V8 修正] ***: 這裡的計算自動包含了 'forward_lean_offset'
    # 腳踝會自動補償臀部的前傾，使機器人傾斜
    l_shin_ankle = amp_hip_pitch * math.sin(phase_left_leg-math.pi*1) + forward_lean_offset
    r_shin_ankle = amp_hip_pitch * math.sin(phase_left_leg-math.pi*1) + forward_lean_offset
    motors["l_shin_l_ankle"].setPosition(l_shin_ankle)
    motors["r_shin_r_ankle"].setPosition(r_shin_ankle)

    # 6. 手臂擺動 (Arm Swing)
    motors["torso_l_shoulder"].setPosition(amp_arm_swing * math.sin(phase_right_leg))
    motors["torso_r_shoulder"].setPosition(amp_arm_swing * math.sin(phase_left_leg))

    # 7. 將未使用的馬達設定為中立位置 (0)
    motors["lb_servo_l_hip"].setPosition(0.0)
    motors["rb_servo_r_hip"].setPosition(0.0)
    motors["l_shoulder_ls_servo"].setPosition(0.0)
    motors["r_shoulder_rs_servo"].setPosition(0.0)
    motors["le_servo_l_elbow"].setPosition(0.0) 
    motors["re_servo_r_elbow"].setPosition(0.0)