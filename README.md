# PLEN2 Webots RL Walk Training

![Webots](https://img.shields.io/badge/Webots-R2023b%2B-blue.svg) ![Python](https://img.shields.io/badge/Python-3.8%2B-yellow.svg) ![SB3](https://img.shields.io/badge/RL-Stable%20Baselines3-green.svg)

本專案使用 **Reinforcement Learning (PPO)** 在 Webots 模擬環境中訓練 PLEN2 機器人
重點在於縮小 **Sim-to-Real (模擬與現實)** 的差距，透過神經網路模擬真實伺服馬達 (MG90s) 的動態特性，並加入隨機推力來訓練機器人的抗干擾能力。

## 檔案結構與功能

本專案由三個核心 Python 檔案組成：

### 1. `train_main.py` 
* **功能**：負責建立 PPO 模型並開始訓練流程。
* **設定**：
  * 使用 **External Control** (`ipc://1234/MyPlenBot`) 連接 Webots。
  * 設定訓練總步數為 **2億步 (200M steps)**。
  * 每 **100,000 步** 自動儲存模型 Checkpoint 至 `./models/`。
  * 支援 TensorBoard Log 紀錄 (`./ppo_plen_teacher_logs/`)。

### 2. `webots_gym_env.py` 
* **功能**：定義符合 Gymnasium 介面的 Webots 環境 (`PlenWalkEnv`)。
* **狀態空間 (Observation)**：49 維，包含 RPY、角速度、重力向量、關節位置/速度、上一次動作、摩擦係數與推力向量。
* **動作空間 (Action)**：12 個伺服馬達的控制訊號。
* **特色機制**：
  * **Push Logic**：每 4 秒施加隨機方向的推力，並包含「推力後穩定期 (Post-Push)」檢測，強化站立穩定性。
  * **Domain Randomization**：隨機化機器人各部件質量與地面摩擦係數。
  * **Reset 流程**：空中收腿 -> 落地 -> 暖身 (Warmup)，確保初始狀態穩定。

### 3. `actuator_net.py` (致動器模擬)
* **功能**：模擬真實 MG90s 伺服馬達的延遲與動態反應。
* **技術細節**：
  * 讀取預訓練的 `.pth` 模型權重。
  * **純 NumPy 推理**：為了加速模擬效率，將 PyTorch 權重轉換為 NumPy 矩陣運算，移除 PyTorch 的 Runtime 開銷。
  * 架構：`Linear -> Tanh -> Linear -> Tanh -> Linear`。

---

## 環境需求

請確保安裝以下套件：
```bash
pip install gymnasium stable-baselines3 shimmy numpy torch tensorboard
```

1.  **Webots Simulator** (建議 R2025a 或更新版本)
2.  **Actuator 模型權重檔**：請確保目錄下有 `mg90s_actuator_net_optimized.pth` (由 `webots_gym_env.py` 載入)。

-----

## 如何執行訓練

### 步驟 1: 設定 Webots 場景

1.  開啟 Webots 並載入 PLEN2 的世界檔。
2.  找到機器人節點，將 `controller` 欄位設定為 `<extern>`。
3.  (選用) 為了配合 `train_main.py` 的設定，你可能需要在 Webots 啟動參數中指定 IPC 通道，或保持預設。

### 步驟 2: 執行訓練腳本

在終端機中執行：

```bash
python train_main.py
```

程式會自動尋找 `WEBOTS_HOME` 並嘗試透過 `ipc://1234/MyPlenBot` 連接模擬器。

### 步驟 3: 監控訓練進度

開啟另一個終端機查看 TensorBoard：

```bash
tensorboard --logdir ./ppo_plen_teacher_logs/
```
