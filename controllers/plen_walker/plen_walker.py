"""
Webots Python 控制器 (CPG 步行)
適用於 MyPlenBot.proto
"""

from controller import Robot
import math

# --- 1. 初始化 ---
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# --- 2. 取得馬達裝置 ---
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

# --- 3. 設定 CPG 參數 ---
walk_freq = 4.0       # 步行速度 (rad/s)

# 振幅 (Amplitude)
amp_hip_pitch = 0.3     # 大腿前後
amp_knee_pitch = 0.3    # 膝蓋彎曲
amp_hip_roll = 0.15      # 臀部左右 (重心轉移)
amp_ankle_roll = 0.2   # 腳踝左右
amp_arm_swing = 0.0     # [修正] 手臂不動

# 前傾偏移量 (Forward Lean Offset)
forward_lean_offset = 0

# --- 4. 主迴圈 ---
print("啟動 CPG 步行控制器 (v10 手臂靜止 + 步態還原)...")

while robot.step(timestep) != -1:
    time = robot.getTime()

    # --- 4.1 相位生成 ---
    # [還原] 左右腳相位改回相同 (+math.pi)
    # 對於鏡像安裝的機器人，相同相位 = 一前一後 (走路)
    phase_left_leg = time * walk_freq - math.pi
    phase_right_leg = time * walk_freq - math.pi  
    
    # 重心轉移相位
    phase_weight_shift = time * walk_freq + math.pi / 2

    # --- 4.2 計算目標角度 ---

    # 1. 大腿前後 (Hip Pitch)
    l_hip_pitch = amp_hip_pitch * math.sin(phase_left_leg - math.pi*0) + forward_lean_offset
    r_hip_pitch = amp_hip_pitch * math.sin(phase_right_leg - math.pi*0) + forward_lean_offset
    
    motors["l_thigh_l_knee"].setPosition(l_hip_pitch)
    motors["r_thigh_r_knee"].setPosition(r_hip_pitch)

    # 2. 膝蓋彎曲 (Knee Pitch)
    l_knee = amp_knee_pitch * max(0, math.sin(phase_left_leg + math.pi / 4))
    r_knee = amp_knee_pitch * max(0, math.sin(phase_right_leg + math.pi / 4))
    
    motors["l_knee_l_shin"].setPosition(l_knee)
    motors["r_knee_r_shin"].setPosition(r_knee)

    # 3. 臀部左右 (Hip Roll)
    hip_roll = amp_hip_roll * math.sin(phase_weight_shift)
    motors["l_hip_l_thigh"].setPosition(hip_roll)
    motors["r_hip_r_thigh"].setPosition(hip_roll)

    # 4. 腳踝左右 (Ankle Roll)
    ankle_roll = amp_ankle_roll * math.sin(phase_weight_shift)
    motors["l_ankle_l_foot"].setPosition(ankle_roll)
    motors["r_ankle_r_foot"].setPosition(ankle_roll) 

    # 5. 腳踝前後 (Ankle Pitch) - [保持幾何修正]
    # 使用負號 (-amp_hip_pitch) 來抵消大腿角度，保持腳掌平行地面
    l_hip_target = l_hip_pitch 
    l_knee_target = l_knee
    r_hip_target = r_hip_pitch
    r_knee_target = r_knee
    motors["l_shin_l_ankle"].setPosition( 1 * (l_hip_target + l_knee_target) - forward_lean_offset )
    motors["r_shin_r_ankle"].setPosition( 1 * (r_hip_target + r_knee_target) - forward_lean_offset )
    
    # 6. 手臂擺動 (Arm Swing) - [修正] 固定為 0
    motors["torso_l_shoulder"].setPosition(0.0)
    motors["torso_r_shoulder"].setPosition(0.0)

    # 7. 其他未使用的馬達歸零
    motors["lb_servo_l_hip"].setPosition(0.0)
    motors["rb_servo_r_hip"].setPosition(0.0)
    motors["l_shoulder_ls_servo"].setPosition(0.0)
    motors["r_shoulder_rs_servo"].setPosition(0.0)
    motors["le_servo_l_elbow"].setPosition(0.0) 
    motors["re_servo_r_elbow"].setPosition(0.0)