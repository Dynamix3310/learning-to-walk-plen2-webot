import os
import sys
import gc

# --- Webots 路徑設定 ---
if 'WEBOTS_HOME' not in os.environ:
    default_path = r"C:\Program Files\Webots"
    if os.path.exists(default_path):
        os.environ['WEBOTS_HOME'] = default_path
    else:
        print("錯誤: 找不到 WEBOTS_HOME，請確認安裝路徑。")
        sys.exit(1)

webots_python_path = os.path.join(os.environ['WEBOTS_HOME'], 'lib', 'controller', 'python')
if webots_python_path not in sys.path:
    sys.path.append(webots_python_path)
# ---------------------

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import torch
from controller import Supervisor
from actuator_net import ActuatorModel

class PlenWalkEnv(gym.Env):
    def __init__(self):
        super(PlenWalkEnv, self).__init__()
        
        # 初始化 Supervisor
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.robot_node = self.robot.getFromDef("MyPlenBot")
        if self.robot_node is None: self.robot_node = self.robot.getSelf()
        
        # 獲取位置與旋轉的 Field，並記錄初始狀態以便手動 Reset
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")
        self.initial_translation = self.trans_field.getSFVec3f()
        self.initial_rotation = self.rot_field.getSFRotation()

        # 載入 ActuatorNet
        try:
            self.actuator = ActuatorModel('mg90s_actuator_net_optimized.pth')
        except Exception as e:
            print(f"ActuatorNet 載入失敗: {e}")
            sys.exit(1)
            
        self.motor_names = [
            "lb_servo_l_hip", "l_hip_l_thigh", "l_thigh_l_knee", "l_knee_l_shin", "l_shin_l_ankle", "l_ankle_l_foot",
            "rb_servo_r_hip", "r_hip_r_thigh", "r_thigh_r_knee", "r_knee_r_shin", "r_shin_r_ankle", "r_ankle_r_foot"
        ]
        self.motors = []
        self.motor_limits = []
        self.SERVO_OFFSET = 70.0
        SAFETY_MARGIN = 1e-3

        print("\n--- 初始化馬達 ---")
        for name in self.motor_names:
            motor = self.robot.getDevice(name)
            if motor is None:
                print(f"嚴重錯誤: 找不到馬達 {name}")
                sys.exit(1)
            motor.enableTorqueFeedback(self.timestep)
            
            # [重要] 確保初始化時馬達速度設為最大，以啟用位置控制
            motor.setVelocity(motor.getMaxVelocity())
            
            self.motors.append(motor)
            min_pos = motor.getMinPosition()
            max_pos = motor.getMaxPosition()
            if min_pos == 0 and max_pos == 0: min_pos, max_pos = -1.57, 1.57
            self.motor_limits.append((min_pos + SAFETY_MARGIN, max_pos - SAFETY_MARGIN))
        print("--------------------------------\n")

        # Sensors
        self.imu = self.robot.getDevice("imu/data inertial")
        self.gyro = self.robot.getDevice("imu/data gyro")
        self.acc = self.robot.getDevice("imu/data accelerometer")
        if self.imu is None: sys.exit(1)
        self.imu.enable(self.timestep)
        self.gyro.enable(self.timestep)
        self.acc.enable(self.timestep)
        
        # 記錄各部位的「初始質量」
        self.num_motors = len(self.motors)
        self.mass_fields = {}
        self.initial_masses = {} 
        
        body_parts = ["TORSO", "R_HIP_LINK", "R_THIGH_LINK", "R_SHIN_LINK", "R_FOOT_LINK", "L_HIP_LINK", "L_THIGH_LINK", "L_SHIN_LINK", "L_FOOT_LINK"]
        for part in body_parts:
            node = self.robot.getFromDef(part)
            if node: 
                field = node.getField("physics").getSFNode().getField("mass")
                self.mass_fields[part] = field
                self.initial_masses[part] = field.getFloat()

        # Spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_motors,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + 12 + 12 + 12 + 1 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # State Init
        self.hist_real = np.full((self.num_motors, self.actuator.hist_len), self.SERVO_OFFSET)
        self.hist_cmd = np.full((self.num_motors, self.actuator.cmd_len), self.SERVO_OFFSET)
        self.current_real_pos = np.zeros(self.num_motors)
        self.prev_action = np.zeros(self.num_motors)
        self.current_friction = 1.0
        
        # Push Logic
        self.PUSH_INTERVAL_SEC = 4.0
        self.PUSH_DURATION_SEC = 0.4
        self.MAX_FORCE_MAGNITUDE = 0.4
        self.push_interval_steps = int(self.PUSH_INTERVAL_SEC * 1000 / self.timestep)
        self.push_duration_steps = int(self.PUSH_DURATION_SEC * 1000 / self.timestep)
        self.push_force = np.zeros(3)
        self.current_push_steps = 0
        self.post_push_steps = 0 # [新增] 推力結束後的穩定期計數器
        self.current_force_vec = [0, 0, 0]
        self.step_count = 0

    def _get_projected_gravity(self, rpy):
        roll, pitch, yaw = rpy
        gx = math.sin(pitch)
        gy = -math.sin(roll) * math.cos(pitch)
        gz = -math.cos(roll) * math.cos(pitch)
        return np.array([gx, gy, gz])

    def step(self, action):
        self.step_count += 1
        
        # Action Remapping
        target_rads = []
        for i, (min_pos, max_pos) in enumerate(self.motor_limits):
            a = np.clip(action[i], -1.0, 1.0)
            if a >= 0: target = a * max(0.0, max_pos)
            else: target = a * abs(min(0.0, min_pos))
            target_rads.append(target)
        target_rads = np.array(target_rads)
        target_deg = np.degrees(target_rads)

        # ActuatorNet Prediction
        self.hist_cmd = np.roll(self.hist_cmd, 1, axis=1)
        self.hist_cmd[:, 0] = target_deg + self.SERVO_OFFSET
        servo_real_deg = self.actuator.predict(self.hist_real, self.hist_cmd)
        self.hist_real = np.roll(self.hist_real, 1, axis=1)
        self.hist_real[:, 0] = servo_real_deg 
        webots_real_deg = servo_real_deg - self.SERVO_OFFSET
        webots_real_rad = np.radians(webots_real_deg)
        
        # Apply to Webots
        for i, (min_pos, max_pos) in enumerate(self.motor_limits):
            webots_real_rad[i] = np.clip(webots_real_rad[i], min_pos, max_pos)
            self.motors[i].setPosition(webots_real_rad[i])
        self.current_real_pos = webots_real_rad

        # --- Push Logic (修改版) ---
        self.push_force = np.zeros(3)
        
        # 1. 觸發新的推力
        if self.step_count > 0 and self.step_count % self.push_interval_steps == 0:
            angle = np.random.uniform(0, 2 * np.pi)
            force_mag = np.random.uniform(0.1, self.MAX_FORCE_MAGNITUDE)
            self.current_force_vec = [force_mag * np.cos(angle),force_mag * np.sin(angle), 0]
            self.current_push_steps = self.push_duration_steps
            self.post_push_steps = 0 # 新推力開始，重置後續穩定計時
            
        # 2. 執行推力
        if self.current_push_steps > 0:
            self.robot_node.addForce(self.current_force_vec, False)
            self.push_force = np.array(self.current_force_vec)
            self.current_push_steps -= 1
            self.robot.setLabel(1, f"⚠️ PUSH! {self.current_push_steps}", 0.05, 0.1, 0.1, 0xff0000, 0.0, "Arial")
            
            # [新增] 當推力剛好結束的瞬間，開啟「後續穩定期 (Post-Push)」
            if self.current_push_steps == 0:
                # 設定 1.5 秒的穩定期，這段時間若不倒，獎勵照算
                self.post_push_steps = int(1.5 * 1000 / self.timestep) 
        
        # 3. 處理後續穩定期 (Post-Push)
        elif self.post_push_steps > 0:
            self.robot.setLabel(1, f"✅ Stabilizing {self.post_push_steps}", 0.05, 0.1, 0.1, 0x00ff00, 0.0, "Arial")
            self.post_push_steps -= 1
        else:
            self.robot.setLabel(1, "", 0, 0, 0, 0, 0, "Arial")

        self.robot.step(self.timestep)
        
        # Observations
        rpy = self.imu.getRollPitchYaw()
        gyro = self.gyro.getValues()
        grav = self._get_projected_gravity(rpy)
        joint_vel = (webots_real_rad - self.current_real_pos) / (self.timestep / 1000.0)
        
        obs = np.concatenate([rpy, gyro, grav, self.current_real_pos, joint_vel, self.prev_action, [self.current_friction], self.push_force]).astype(np.float32)
        
        # --- Reward Function (修正版) ---
        vel = self.robot_node.getVelocity()
        current_z = self.robot_node.getPosition()[2]
        
        # 判定是否站立
        is_standing = (current_z > -0.13) and (abs(rpy[0]) < 1.0) and (abs(rpy[1]) < 1.0)
        
        # 1. 存活獎勵
        r_alive = 10.0 if is_standing else 0.0
        
        # 2. 速度懲罰(暫時停用)
        r_vel = 0 * (vel[0]**2 + vel[1]**2)
        
        # 3. 穩定性懲罰
        r_stable = -0.2 * (abs(rpy[0]) + abs(rpy[1]))
        
        # 4.姿勢懲罰 (Pose Penalty)
        r_pose = -3 * np.sum(np.square(self.current_real_pos))

        # 5. 平滑度懲罰 (Smoothness)
        r_smooth = -0.5 * np.sum(np.square(action - self.prev_action))
        
        # 6. 抗推力獎勵 (包含推力期間 + 推力結束後的穩定期)
        is_under_pressure = (self.current_push_steps > 0) or (self.post_push_steps > 0)
        r_resist = 5.0 if is_under_pressure and is_standing else 0.0

        # 7.懲罰偏離正面
        r_facing = -1 * abs(rpy[2])

        #8.位置偏移懲罰
        current_pos = self.robot_node.getPosition()
        dx = current_pos[0] - self.initial_translation[0]
        dy = current_pos[1] - self.initial_translation[1]
        r_pos_drift = -1.5 * (dx**2 + dy**2)

        reward = r_alive + r_vel + r_stable + r_pose + r_smooth + r_resist + r_facing + r_pos_drift

        terminated = False
        truncated = False
        if current_z < -0.13 or abs(rpy[0]) > 1.0 or abs(rpy[1]) > 1.0: 
            terminated = True
            reward -= 10.0
            
        self.prev_action = action
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        gc.collect() 
        super().reset(seed=seed)
        
        # --- 階段一：空中收腿 (Air Reset) ---
        air_pos = [self.initial_translation[0], self.initial_translation[1], self.initial_translation[2] + 0.05]
        self.trans_field.setSFVec3f(air_pos)
        self.rot_field.setSFRotation(self.initial_rotation)
        self.robot_node.resetPhysics() 
        
        for motor in self.motors:
            motor.setVelocity(motor.getMaxVelocity()) 
            motor.setPosition(0.0) 
            
        air_steps = int(0.2 * 1000 / self.timestep)
        for _ in range(air_steps):
            self.robot.step(self.timestep)
            
        # --- 階段二：落地與隨機化 (Ground Reset) ---
        self.trans_field.setSFVec3f(self.initial_translation)
        self.rot_field.setSFRotation(self.initial_rotation)
        self.robot_node.resetPhysics() 
        
        for name, field in self.mass_fields.items():
            base_mass = self.initial_masses[name]
            field.setFloat(base_mass * np.random.uniform(0.8, 1.2))

        # --- 階段三：地面暖身 (Warmup) ---
        warmup_seconds = 1.0
        warmup_steps = int(warmup_seconds * 1000 / self.timestep)
        
        for _ in range(warmup_steps):
            for motor in self.motors:
                motor.setPosition(0.0) 
            self.robot.step(self.timestep)
        
        # --- 階段四：初始化狀態變數 ---
        self.hist_real.fill(self.SERVO_OFFSET)
        self.hist_cmd.fill(self.SERVO_OFFSET)
        self.prev_action.fill(0)
        self.current_real_pos.fill(0)
        self.current_friction = np.random.uniform(0.4, 1.2)
        self.push_force = np.zeros(3)
        self.current_push_steps = 0
        self.post_push_steps = 0 # [重要] Reset 時也要歸零
        self.step_count = 0
        
        self.robot.step(self.timestep)
        
        rpy = self.imu.getRollPitchYaw()
        gyro = self.gyro.getValues()
        grav = self._get_projected_gravity(rpy)
        
        obs = np.concatenate([rpy, gyro, grav, np.zeros(12), np.zeros(12), np.zeros(12), [self.current_friction], [0,0,0]]).astype(np.float32)
        return obs, {}