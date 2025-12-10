import os
import sys

# --- 自動設定 Webots 路徑 (防止 import 錯誤) ---
if 'WEBOTS_HOME' not in os.environ:
    default_path = r"C:\Program Files\Webots"
    if os.path.exists(default_path):
        os.environ['WEBOTS_HOME'] = default_path
    else:
        print("錯誤: 找不到 WEBOTS_HOME，請確認安裝路徑。")

webots_python_path = os.path.join(os.environ['WEBOTS_HOME'], 'lib', 'controller', 'python')
if webots_python_path not in sys.path:
    sys.path.append(webots_python_path)

# 設定外部控制器連線目標 (必須與 Webots Console 顯示的一致)
os.environ['WEBOTS_CONTROLLER_URL'] = 'ipc://1234/MyPlenBot'
# ------------------------------------------------------------------

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from webots_gym_env import PlenWalkEnv
import numpy as np
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.reward_stats = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        
        if infos:
            info = infos[0]
            
            for key, value in info.items():
                if key.startswith("rewards/"):
                    self.logger.record(key, value)
                    if key not in self.reward_stats:
                        self.reward_stats[key] = {"max": -np.inf, "min": np.inf}
                    
                    self.reward_stats[key]["max"] = max(self.reward_stats[key]["max"], value)
                    self.reward_stats[key]["min"] = min(self.reward_stats[key]["min"], value)

        return True
        
    def _on_rollout_end(self) -> None:
        print("\n--- Reward Statistics (Rollout End) ---")
        for key, stats in self.reward_stats.items():
            print(f"{key}: Min={stats['min']:.4f}, Max={stats['max']:.4f}")
        print("---------------------------------------")
        
def main():
    print("正在初始化 Webots 環境...")
    env = PlenWalkEnv()
    print("環境初始化成功！準備開始訓練...")

    reward_callback = RewardLoggerCallback()
    # 每 100,000 步儲存一次模型到 ./models/ 資料夾
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='./models/',
        name_prefix='ppo_plen_model'
    )
    callbacks = [checkpoint_callback, reward_callback]
    # 定義 PPO Teacher Policy
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])] 
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(3e-4),
        n_steps=4096,
        batch_size=256,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_plen_teacher_logs/"
    )
    
    print("開始訓練 Teacher Policy (總計 2000萬步)...")
    try:
        model.learn(
            total_timesteps=20000000, 
            callback=callbacks # 加入 callback
        )
        
        # 訓練結束儲存最終版
        model.save("ppo_plen_teacher_final")
        print("訓練完成！模型已儲存為 ppo_plen_teacher_final.zip")
        
    except KeyboardInterrupt:
        print("\n訓練被手動中斷！正在緊急儲存模型...")
        model.save("ppo_plen_teacher_interrupted")
        print("模型已儲存為 ppo_plen_teacher_interrupted.zip")

if __name__ == "__main__":
    main()

#cd C:\Users\DYNAMIX\Desktop\+\webot_PLEN_model_improve\train_main
#tensorboard --logdir ./ppo_plen_teacher_logs/