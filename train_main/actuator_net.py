import torch
import numpy as np
import os

class ActuatorModel:
    def __init__(self, model_path, device="cpu"):
        # 雖然這裡用到 torch，但只用於讀取權重
        # 讀取完後我們就會把 torch 相關的東西丟掉
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        print(f"Loading ActuatorNet from {model_path} (Converting to Pure Numpy)...")
        
        # 讀取 .pth
        loaded_data = torch.load(model_path, map_location="cpu", weights_only=False)
        
        self.hist_len = loaded_data['actuator_history_len']
        self.cmd_len = loaded_data['cmd_history_len']
        self.input_dim = self.hist_len + self.cmd_len
        
        # 讀取標準化參數 (轉為 numpy)
        self.mean_X = loaded_data['mean_X']
        self.std_X = loaded_data['std_X']
        self.mean_Y = loaded_data['mean_Y']
        self.std_Y = loaded_data['std_Y']
        
        # 提取權重與偏差 (Weights & Biases)
        state_dict = loaded_data['model_state_dict']
        
        # Layer 1
        self.w1 = state_dict['net.0.weight'].numpy().T # Transpose for matrix mul
        self.b1 = state_dict['net.0.bias'].numpy()
        
        # Layer 2
        self.w2 = state_dict['net.2.weight'].numpy().T
        self.b2 = state_dict['net.2.bias'].numpy()
        
        # Layer 3 (Output)
        self.w3 = state_dict['net.4.weight'].numpy().T
        self.b3 = state_dict['net.4.bias'].numpy()
        
        print("ActuatorNet 已轉換為 Numpy 矩陣運算模式 (無 PyTorch 開銷)")

    def predict(self, real_hist, cmd_hist):
        """
        純 Numpy 前向傳播 (Forward Pass)
        架構: Linear -> Tanh -> Linear -> Tanh -> Linear
        """
        # 1. Concat
        features = np.concatenate([real_hist, cmd_hist], axis=1)
        
        # 2. Normalize
        x = (features - self.mean_X) / self.std_X
        
        # 3. Layer 1: x @ w1 + b1 -> Tanh
        h1 = np.tanh(np.dot(x, self.w1) + self.b1)
        
        # 4. Layer 2: h1 @ w2 + b2 -> Tanh
        h2 = np.tanh(np.dot(h1, self.w2) + self.b2)
        
        # 5. Layer 3 (Output): h2 @ w3 + b3
        out_norm = np.dot(h2, self.w3) + self.b3
        
        # 6. De-normalize
        out_deg = out_norm.flatten() * self.std_Y + self.mean_Y
        
        return out_deg
