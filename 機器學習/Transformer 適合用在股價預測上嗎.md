---
title: Transformer 適合用在股價預測上嗎?

---

{%preview https://www.sciencedirect.com/science/article/abs/pii/S0957417423019267 %}
Series decomposition Transformer with period-correlation for stock market index prediction
這篇論文很短，主要就是模型架構跟演算法怎麼跑的。
模型創新：提出了一種名為 SDTP 的新型深度學習模型。
核心機制：該模型結合了兩大關鍵技術：
* 序列分解層 (Series Decomposition Layer)。
* 週期相關機制 (Period-Correlation Mechanism)。

目的：透過這些機制來捕捉時間序列資料中固有的週期性 (Inherent Periodicity) 以及序列之間的關聯性 (Relation)。

下面是我做的PPT，懶得轉成文字了
![image](https://hackmd.io/_uploads/HJPR6mPB-g.png)
![image](https://hackmd.io/_uploads/SJrgauwB-e.png)
![image](https://hackmd.io/_uploads/ByMJ6_vB-l.png)
![image](https://hackmd.io/_uploads/ryWfTuDB-g.png)
![image](https://hackmd.io/_uploads/HJ17p_PH-l.png)
![image](https://hackmd.io/_uploads/ry6NTdPS-g.png)
![image](https://hackmd.io/_uploads/ryurT_vH-l.png)
![image](https://hackmd.io/_uploads/H1B86uPS-x.png)
![image](https://hackmd.io/_uploads/SJe_a_wrbe.png)
![image](https://hackmd.io/_uploads/Sk1KpdwrZl.png)
![image](https://hackmd.io/_uploads/HkOtaOPSZe.png)
![image](https://hackmd.io/_uploads/H1G9p_PSZg.png)
![image](https://hackmd.io/_uploads/ry59TOvHWe.png)
![image](https://hackmd.io/_uploads/H1Es6uwB-e.png)


----
論文沒給參數跟模型原始碼，我跑S&P的數據做不出來跟論文一樣的結果。
但他這個想法算是有趣，後續可以針對這個模型做些調整
#### 趨勢定義問題
* 使用 AvgPool (移動平均) 來定義「趨勢 ($X_t$)」
    * 移動平均線在本質上是滯後指標
    * 這導致模型訓練時，Encoder 認定「趨勢」還在往下，但實際上價格已經反轉向上
* 修正
    * 將 AvgPool 替換為 Conv1d 層
        * 讓神經網路自己學習什麼樣的平滑曲線最適合代表當下的趨勢
    * 改用多項式擬合
        * 使用 Autoformer 中的做法，利用多項式回歸來提取趨勢
#### Decoder 初始化問題
* 論文在 Decoder 輸入端，將未來的波動部分 ($X_{des}$) 填補為 0
    * 這假設了市場總是傾向於「瞬間回歸均線」，對於 動能強勁 的趨勢股，會導致預測值嚴重低估波動幅度
* 修正
    * 線性外推 
        * 不要補 0，而是計算過去幾點的斜率，將波動值線性延伸填入
    * 最後值填充
        * 最簡單的作法，假設波動維持在最後一刻的強度
    * 
#### 正規化方法的問題
* 論文使用全域統計數據做 Z-score
    * 金融時間序列具有 非平穩性 (Non-stationarity)
    * 2020 年的波動率 (Std) 可能跟 2024 年完全不同，會導致「概念飄移 (Concept Drift)」
* 修正
    * 引入 RevIN (優先處理這個，這個方法有料)
        * 對每一個 Batch 的輸入獨立做正規化，模型只學「形狀」，輸出後再把該 Batch 的平均值與標準差乘回去。這能大幅提升模型對不同市場環境的適應力

#### 損失函數的選擇 (優先處理，我覺得這裡他確實選了一個滿爛的 loss function)
* 使用 MSE (均方誤差) 作為 Loss Function
    * MSE 對於「方向」不敏感，且對異常值過於敏感
    * 在交易中，預測漲跌方向 (Direction) 往往比預測絕對數值更重要
* 修正 
    * 加入方向性懲，給予額外的重罰
    * 不要只預測一個價格，而是預測價格的區間




底下是嘗試去做論文復現的程式碼,進行了兩處修正。
## 程式範例
### 改動可學習序列分解 (Learnable Series Decomposition)
$$W' = \text{Softmax}(W_{learnable})$$
$$X_{trend} = X * W'$$
$$X_{seasonal} = X - X_{trend}$$
為什麼要用 Softmax？
為了保持「趨勢」的物理意義，濾波器的權重總和必須為 1 ($\sum W'_i = 1$)。如果不做限制，卷積後的數值會無限放大或縮小，導致 $X - X_{trend}$ 失去意義。
### 改動動能外推初始化 (Linear Extrapolation Initialization)
令 Encoder 輸出的最後一個時間點為 $t$，倒數第二個時間點為 $t-1$。
對於未來第 $\tau$ 個預測點 ($1 \le \tau \le L_{pred}$)，初始化的預測值 $\hat{X}_{t+\tau}$ 計算如下：
$$\text{Slope} = X_t - X_{t-1}$$
$$\hat{X}_{t+\tau} = X_t + \text{Slope} \times \tau$$
$X_t$ 是 Encoder 輸出的最後一筆數據。
$\text{Slope}$ 代表當下的瞬時速度。
最後的改善後的結果有比原來我進行複現的程式碼表現好，但還是沒有比論文中的高，論文對s&p的準確度高達0.98，我只能做到0.975，真的不曉得參數該怎麼調了

## 程式碼
```
# ==========================================
# 1. 套件安裝與匯入
# ==========================================
# !pip install yfinance scikit-learn matplotlib
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

print(f"JAX 版本: {jax.__version__}")
print(f"裝置: {jax.devices()}")
print()

# ==========================================
# 2. 真實數據載入與處理 (Real World Data)
# ==========================================

# 設定參數 (論文 Table 4 指定 S&P 500 區間)
TICKER = "^GSPC"
START_DATE = "2010-01-04"
END_DATE = "2018-12-28"

print(f"📥 正在下載 {TICKER} 數據...")
# --- 建構論文 Section 5.1 定義的 8 個特徵 ---
# 1. Volume
# 2. Turnover (估算值: Volume * Close，因 Yahoo 不提供指數成交額)
# 3. Change (Close - Prev Close)
# 4. Change rate ((Close - Prev Close) / Prev Close)
# 5. High
# 6. Low
# 7. Open
# 8. Close

df = yf.download(TICKER, start=START_DATE, end=END_DATE)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
# 為了計算 Change 和 Change Rate，我們需要 shift
df['Prev_Close'] = df['Close'].shift(1)

# 計算特徵
df['Change'] = df['Close'] - df['Prev_Close']
df['Change_Rate'] = (df['Close'] - df['Prev_Close']) / df['Prev_Close']
df['Turnover'] = df['Volume'] * df['Close'] # 估算

# 移除第一筆 (因為 shift 產生 NaN)
df = df.dropna()

# 選取並排序特徵 (依照 Table 2 順序)
feature_cols = [
    'Volume', 'Turnover', 'Change', 'Change_Rate',
    'High', 'Low', 'Open', 'Close'
]

# 確保只取這些欄位的值
data_raw = df[feature_cols].values

print(f"✅ 特徵工程完成，資料形狀: {data_raw.shape}")
print(f"   包含特徵: {feature_cols}")


print(f"✅ 下載完成，總筆數: {len(data_raw)}")

# --- 數據正規化 (Z-score) ---
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_raw)

# --- 製作 Time Series Dataset ---
# 依據論文參數: Lag=5 (SEQ_LEN), Predict=1 (PRED_LEN)
SEQ_LEN = 5
PRED_LEN = 1

X_data, Y_data = [], []
for i in range(len(data_scaled) - SEQ_LEN - PRED_LEN+1):
    X_data.append(data_scaled[i : i+SEQ_LEN])      
    Y_data.append(data_scaled[i+SEQ_LEN : i+SEQ_LEN+PRED_LEN]) 

X = np.array(X_data).astype(np.float32)  # Shape: [N, 5, 8]
Y = np.array(Y_data).astype(np.float32)  # Shape: [N, 1, 8]

# 切分訓練集 (80%) 與測試集 (20%)
train_size = int(len(X) * 0.8)
X_train, Y_train = X[:train_size], Y[:train_size]
X_test, Y_test = X[train_size:], Y[train_size:]

print(f"📊 數據集準備完成: 訓練集 {len(X_train)} 筆, 測試集 {len(X_test)} 筆")
print()

# ==========================================
# 3. SDTP 模型定義 (JAX 版本 - 改進版)
# ==========================================

# 參數設定
INPUT_DIM = 8
D_MODEL = 64
N_HEADS = 4
N_ENCODER_LAYERS = 2
N_DECODER_LAYERS = 2
D_FF = 256
KERNEL_SIZE = 3
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# ============================================================
# 核心組件 (改進版：可學習分解 + 動能外推)
# ============================================================

def linear_extrapolation(seq, pred_len):
    """
    [NEW] 線性動能外推
    seq: (batch, seq_len, features)
    """
    # 取最後兩點計算 "速度" (Slope)
    last_val = seq[:, -1:, :]
    prev_val = seq[:, -2:-1, :]

    # 斜率 = (現在 - 上一刻)
    slope = last_val - prev_val

    # 產生未來的時間步長 [1, 2, ..., pred_len]
    time_steps = jnp.arange(1, pred_len + 1).reshape(1, -1, 1)

    # 未來預測 = 最後一點 + 斜率 * 時間
    pred = last_val + slope * time_steps
    return pred

def learnable_moving_average(x, kernel_weights):
    """
    [NEW] 使用可學習權重的移動平均
    x: (seq_len,)
    kernel_weights: (kernel_size,)
    """
    kernel_size = kernel_weights.shape[0]
    pad_size = kernel_size // 2

    # Padding
    x_padded = jnp.pad(x, pad_size, mode='edge')

    # 關鍵：使用 softmax 確保權重總和為 1 (保持趨勢的數值規模)
    w_norm = jax.nn.softmax(kernel_weights)

    # 卷積
    trend = jnp.convolve(x_padded, w_norm, mode='valid')

    # 處理偶數 kernel 可能導致的長度誤差 (防呆)
    return trend[:x.shape[0]]

def series_decomposition(x, kernel_weights):
    """
    [MODIFIED] 序列分解: Trend + Seasonal
    現在接受 kernel_weights 而不是固定的 kernel_size
    """
    batch, seq_len, features = x.shape

    def process_single(x_single):
        # x_single: (seq_len, features)
        # vmap over features (axis 1)
        return vmap(learnable_moving_average, in_axes=(1, None), out_axes=1)(
            x_single, kernel_weights
        )

    trend = vmap(process_single)(x)
    seasonal = x - trend
    return seasonal, trend

def period_correlation(params, query, key, value, mask=None):
    """Period-Correlation Mechanism"""
    batch_size, seq_len, d_model = query.shape
    d_k = d_model // N_HEADS

    Q = query @ params['W_q']
    K = key @ params['W_k']
    V = value @ params['W_v']

    Q = Q.reshape(batch_size, seq_len, N_HEADS, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, N_HEADS, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, N_HEADS, d_k).transpose(0, 2, 1, 3)

    scores = (Q @ jnp.swapaxes(K, -2, -1)) / jnp.sqrt(d_k)

    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)

    attn = jax.nn.softmax(scores, axis=-1)
    output = attn @ V

    output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    output = output @ params['W_o']

    return output

@jit
def feed_forward(params, x):
    """Feed Forward Network"""
    hidden = jax.nn.relu(x @ params['W1'] + params['b1'])
    output = hidden @ params['W2'] + params['b2']
    return output

@jit
def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer Normalization"""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / jnp.sqrt(var + eps) + beta

def encoder_layer_forward(params, x, kernel_weights):
    """Encoder Layer (Accepts kernel_weights)"""
    # Period-Correlation
    attn_out = period_correlation(params['attn'], x, x, x)
    x = x + attn_out

    # Decomposition
    seasonal, _ = series_decomposition(x, kernel_weights)
    seasonal = layer_norm(seasonal, params['norm1_gamma'], params['norm1_beta'])

    # Feed Forward
    ffn_out = feed_forward(params['ffn'], seasonal)
    seasonal = seasonal + ffn_out

    # Decomposition
    seasonal_out, _ = series_decomposition(seasonal, kernel_weights)
    seasonal_out = layer_norm(seasonal_out, params['norm2_gamma'], params['norm2_beta'])

    return seasonal_out

def decoder_layer_forward(params, seasonal_input, trend_input, enc_output, kernel_weights):
    """Decoder Layer (Accepts kernel_weights)"""
    trend_accum = trend_input

    # Self-Attention
    self_attn = period_correlation(params['self_attn'],
                                   seasonal_input, seasonal_input, seasonal_input)
    seasonal = seasonal_input + self_attn
    seasonal, trend1 = series_decomposition(seasonal, kernel_weights)
    trend_accum = trend_accum + trend1 @ params['W_trend1']

    # Cross-Attention
    cross_attn = period_correlation(params['cross_attn'],
                                   seasonal, enc_output, enc_output)
    seasonal = seasonal + cross_attn
    seasonal, trend2 = series_decomposition(seasonal, kernel_weights)
    trend_accum = trend_accum + trend2 @ params['W_trend2']

    # Feed Forward
    ffn_out = feed_forward(params['ffn'], seasonal)
    seasonal = seasonal + ffn_out
    seasonal_out, trend3 = series_decomposition(seasonal, kernel_weights)
    trend_out = trend_accum + trend3 @ params['W_trend3']

    return seasonal_out, trend_out

# ============================================================
# 模型初始化與前向傳播
# ============================================================

def init_sdtp_params(key,input_dim):
    """初始化 SDTP 模型參數"""
    keys = random.split(key, 25)

    params = {
        'input_proj': random.normal(keys[0], (input_dim, D_MODEL)) * 0.02, # <--- 改這裡
        'output_proj': random.normal(keys[1], (D_MODEL, input_dim)) * 0.02, # <--- 改這裡

        # [NEW] 可學習分解卷積核 (Learnable Kernel)
        # 初始化為 1/k (平均值) 並加上微小雜訊以便梯度下降開始運作
        'decomp_kernel': jnp.ones(KERNEL_SIZE) / KERNEL_SIZE + \
                         random.normal(keys[2], (KERNEL_SIZE,)) * 0.001,

        'encoder': [],
        'decoder': []
    }

    # Encoder
    for i in range(N_ENCODER_LAYERS):
        key_i = keys[3 + i]
        k1, k2, k3, k4, k5, k6 = random.split(key_i, 6)

        encoder_params = {
            'attn': {
                'W_q': random.normal(k1, (D_MODEL, D_MODEL)) * 0.02,
                'W_k': random.normal(k2, (D_MODEL, D_MODEL)) * 0.02,
                'W_v': random.normal(k3, (D_MODEL, D_MODEL)) * 0.02,
                'W_o': random.normal(k4, (D_MODEL, D_MODEL)) * 0.02,
            },
            'ffn': {
                'W1': random.normal(k5, (D_MODEL, D_FF)) * 0.02,
                'b1': jnp.zeros(D_FF),
                'W2': random.normal(k6, (D_FF, D_MODEL)) * 0.02,
                'b2': jnp.zeros(D_MODEL),
            },
            'norm1_gamma': jnp.ones(D_MODEL),
            'norm1_beta': jnp.zeros(D_MODEL),
            'norm2_gamma': jnp.ones(D_MODEL),
            'norm2_beta': jnp.zeros(D_MODEL),
        }
        params['encoder'].append(encoder_params)

    # Decoder
    for i in range(N_DECODER_LAYERS):
        key_i = keys[3 + N_ENCODER_LAYERS + i]
        k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13 = random.split(key_i, 13)

        decoder_params = {
            'self_attn': {
                'W_q': random.normal(k1, (D_MODEL, D_MODEL)) * 0.02,
                'W_k': random.normal(k2, (D_MODEL, D_MODEL)) * 0.02,
                'W_v': random.normal(k3, (D_MODEL, D_MODEL)) * 0.02,
                'W_o': random.normal(k4, (D_MODEL, D_MODEL)) * 0.02,
            },
            'cross_attn': {
                'W_q': random.normal(k5, (D_MODEL, D_MODEL)) * 0.02,
                'W_k': random.normal(k6, (D_MODEL, D_MODEL)) * 0.02,
                'W_v': random.normal(k7, (D_MODEL, D_MODEL)) * 0.02,
                'W_o': random.normal(k8, (D_MODEL, D_MODEL)) * 0.02,
            },
            'ffn': {
                'W1': random.normal(k9, (D_MODEL, D_FF)) * 0.02,
                'b1': jnp.zeros(D_FF),
                'W2': random.normal(k10, (D_FF, D_MODEL)) * 0.02,
                'b2': jnp.zeros(D_MODEL),
            },
            'W_trend1': random.normal(k11, (D_MODEL, D_MODEL)) * 0.01,
            'W_trend2': random.normal(k12, (D_MODEL, D_MODEL)) * 0.01,
            'W_trend3': random.normal(k13, (D_MODEL, D_MODEL)) * 0.01,
        }
        params['decoder'].append(decoder_params)

    return params

def sdtp_forward(params, x, kernel_size=3):
    """SDTP 前向傳播 - 包含動能外推與可學習分解"""
    batch_size = x.shape[0]

    # 提取學習到的 kernel weights
    kernel_weights = params['decomp_kernel']

    # Input Embedding + Decomposition
    x_embed = x @ params['input_proj']
    enc_seasonal, enc_trend = series_decomposition(x_embed, kernel_weights)

    # Encoder
    for enc_params in params['encoder']:
        enc_seasonal = encoder_layer_forward(enc_params, enc_seasonal, kernel_weights)

    # ============================================================
    # [MODIFIED] Decoder Initialization
    # ============================================================

    # 波動軌：歷史 (Token) + 預測 (Prediction)
    # Token 部分
    dec_seasonal_token = enc_seasonal[:, -(SEQ_LEN-PRED_LEN):, :]
    # Prediction 部分：使用 [NEW] 動能外推 而不是補 0
    # 我們對波動項做輕微的外推（或者保持0），這裡示範動能外推
    dec_seasonal_pred = linear_extrapolation(enc_seasonal, PRED_LEN)

    dec_seasonal = jnp.concatenate([dec_seasonal_token, dec_seasonal_pred], axis=1)

    # 趨勢軌：歷史 (Token) + 預測 (Prediction)
    dec_trend_token = enc_trend[:, -(SEQ_LEN-PRED_LEN):, :]
    # Prediction 部分：使用 [NEW] 動能外推 而不是補 Mean
    dec_trend_pred = linear_extrapolation(enc_trend, PRED_LEN)

    dec_trend = jnp.concatenate([dec_trend_token, dec_trend_pred], axis=1)

    # Decoder
    for dec_params in params['decoder']:
        dec_seasonal, dec_trend = decoder_layer_forward(
            dec_params, dec_seasonal, dec_trend, enc_seasonal, kernel_weights
        )

    # Output
    final_seasonal = dec_seasonal[:, -PRED_LEN:, :]
    final_trend = dec_trend[:, -PRED_LEN:, :]
    
    # 【修正】將兩者在隱藏層(64維)先相加，再通過 output_proj 轉回 (8維)
    # 這是最穩健的做法，確保 Trend 和 Seasonal 都經過正確的權重轉換
    predictions = (final_seasonal + final_trend) @ params['output_proj']
    
    return predictions

# ============================================================
# 損失函數與優化器
# ============================================================
@jit
def direction_weighted_loss(params, x, y_true, kernel_size, lambda_dir=5.0):
    """
    結合 MSE 與 方向性懲罰
    lambda_dir: 方向懲罰係數，設越大模型越在意漲跌方向
    """
    # 1. 取得預測值
    y_pred = sdtp_forward(params, x, kernel_size)
    
    # -------------------------------------------------------
    # 技巧：我們不只看數值，更看「變化量 (Delta)」
    # -------------------------------------------------------
    # 取得輸入序列的最後一點 (Last Known Value)
    # x shape: (Batch, Seq, Features), Close is index 7
    last_close = x[:, -1:, 7:8] 
    
    # 計算真實的漲跌 (Delta True)
    # y_true shape: (Batch, Pred, Features)
    delta_true = y_true[:, :, 7:8] - last_close
    
    # 計算預測的漲跌 (Delta Pred)
    delta_pred = y_pred[:, :, 7:8] - last_close
    
    # 2. 基礎 MSE Loss
    mse = jnp.mean((y_pred - y_true) ** 2)
    
    # 3. 方向性 Loss (Directional Loss)
    # 如果 sign(delta_true) != sign(delta_pred)，則給予懲罰
    # jnp.sign 回傳 -1, 0, 1
    true_sign = jnp.sign(delta_true)
    pred_sign = jnp.sign(delta_pred)
    
    # 只有當方向相反時 (相乘 < 0)，才會有值
    direction_error = jnp.where(true_sign * pred_sign < 0, jnp.abs(delta_true - delta_pred), 0.0)
    dir_loss = jnp.mean(direction_error)
    
    # 4. 總 Loss
    total_loss = mse + lambda_dir * dir_loss
    
    return total_loss

def mse_loss(params, x, y_true, kernel_size):
    """MSE Loss - kernel_size 僅作為 padding 參考，實際運算使用 params['decomp_kernel']"""
    y_pred = sdtp_forward(params, x, kernel_size)
    return jnp.mean((y_pred - y_true) ** 2)

# 編譯梯度函數
loss_and_grad = jit(jax.value_and_grad(direction_weighted_loss), static_argnums=(3,))

# 簡單的 Adam 優化器實作
def init_adam_state(params):
    """初始化 Adam 優化器狀態"""
    m = jax.tree.map(lambda p: jnp.zeros_like(p), params)
    v = jax.tree.map(lambda p: jnp.zeros_like(p), params)
    return {'m': m, 'v': v, 't': 0}

@jit
def adam_update(params, grads, opt_state, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    """Adam 優化器更新步驟"""
    t = opt_state['t'] + 1
    m = jax.tree.map(lambda m_i, g: beta1 * m_i + (1 - beta1) * g, opt_state['m'], grads)
    v = jax.tree.map(lambda v_i, g: beta2 * v_i + (1 - beta2) * g**2, opt_state['v'], grads)

    m_hat = jax.tree.map(lambda m_i: m_i / (1 - beta1**t), m)
    v_hat = jax.tree.map(lambda v_i: v_i / (1 - beta2**t), v)

    params = jax.tree.map(
        lambda p, m_i, v_i: p - lr * m_i / (jnp.sqrt(v_i) + eps),
        params, m_hat, v_hat
    )

    return params, {'m': m, 'v': v, 't': t}

# ==========================================
# 4. 訓練迴圈
# ==========================================

print("🚀 開始訓練 SDTP 模型 (JAX 改進版: 動能外推 + 可學習分解)...")
print(f"參數配置: d_model={D_MODEL}, n_heads={N_HEADS}, layers={N_ENCODER_LAYERS}")
print()

# 初始化模型
key = random.PRNGKey(42)
params = init_sdtp_params(key, INPUT_DIM)

# 計算參數量
n_params = sum(x.size for x in jax.tree.leaves(params))
print(f"總參數量: {n_params:,}")
print()

# 初始化優化器
opt_state = init_adam_state(params)

# Warm-up (JIT 編譯)
print("Warm-up (JIT 編譯)...")
x_sample = jnp.array(X_train[:BATCH_SIZE])
y_sample = jnp.array(Y_train[:BATCH_SIZE])

start = time.time()
loss_val, grads = loss_and_grad(params, x_sample, y_sample, KERNEL_SIZE)
jax.tree.map(lambda x: x.block_until_ready(), grads)
t_warmup = time.time() - start
print(f"編譯時間: {t_warmup:.4f}s")
print(f"初始損失: {loss_val:.6f}")
print()

# 訓練循環
loss_history = []
start_time = time.time()

for epoch in range(EPOCHS):
    # 打亂訓練數據
    n_train = len(X_train)
    perm = np.random.permutation(n_train)

    epoch_losses = []

    # 批次訓練
    for i in range(0, n_train, BATCH_SIZE):
        batch_idx = perm[i:i+BATCH_SIZE]
        if len(batch_idx) < BATCH_SIZE:
            continue

        x_batch = jnp.array(X_train[batch_idx])
        y_batch = jnp.array(Y_train[batch_idx])

        # 計算損失和梯度
        loss_val, grads = loss_and_grad(params, x_batch, y_batch, KERNEL_SIZE)

        # Adam 更新
        params, opt_state = adam_update(params, grads, opt_state, lr=LEARNING_RATE)

        epoch_losses.append(float(loss_val))

    avg_loss = np.mean(epoch_losses)
    loss_history.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{EPOCHS}, Loss: {avg_loss:.6f}, Time: {elapsed:.2f}s")

total_time = time.time() - start_time
print()
print(f"✅ 訓練完成！總時間: {total_time:.2f}s")
print()

# ==========================================
# 5. 評估與指標計算 (修正維度錯誤版)
# ==========================================

print("🔍 正在進行測試集評估 (Target: Close Price)...")

# 編譯推論函數
forward_jit = jit(sdtp_forward, static_argnums=(2,))

preds_scaled_all = []
trues_scaled_all = []

# 分批預測
test_batch_size = 32
for i in range(0, len(X_test), test_batch_size):
    x_batch = jnp.array(X_test[i:i+test_batch_size])
    y_batch = Y_test[i:i+test_batch_size]

    pred = forward_jit(params, x_batch, KERNEL_SIZE).block_until_ready()

    # 【關鍵修正 1】提取所有 8 個特徵 (Volume...Close)，而不是只有 Close
    # pred shape: (Batch, 1, 8) -> 取出 (Batch, 8)
    preds_scaled_all.extend(pred[:, 0, :]) 
    trues_scaled_all.extend(y_batch[:, 0, :])

# 轉為 NumPy 陣列，形狀應為 (N, 8)
preds_scaled_all = np.array(preds_scaled_all)
trues_scaled_all = np.array(trues_scaled_all)

# 【關鍵修正 2】反正規化 (現在輸入是 8 維，Scaler 才能正常工作)
preds_real_all = scaler.inverse_transform(preds_scaled_all)
trues_real_all = scaler.inverse_transform(trues_scaled_all)

# 【關鍵修正 3】反正規化後，再單獨取出 Close Price (第 7 欄)
# feature_cols = ['Volume', ..., 'Close']
close_idx = 7 

preds_close = preds_real_all[:, close_idx]
trues_close = trues_real_all[:, close_idx]

# 計算指標
mse = mean_squared_error(trues_close, preds_close)
rmse = np.sqrt(mse)
mae = mean_absolute_error(trues_close, preds_close)
r2 = r2_score(trues_close, preds_close)
mape = np.mean(np.abs((trues_close - preds_close) / trues_close)) * 100

print("-" * 40)
print(f"🏆 測試集評估結果 (S&P 500 Close Price):")
print(f"RMSE (均方根誤差):   {rmse:.4f}")
print(f"MAE  (平均絕對誤差): {mae:.4f}")
print(f"MAPE (百分比誤差):   {mape:.4f}%")
print(f"R²   (決定係數):     {r2:.4f}")
print("-" * 40)
print()

# ==========================================
# 6. 視覺化
# ==========================================

# 預測結果對比
plt.figure(figsize=(12, 6))
plot_len = 100
# 使用修正後的變數名稱: trues_close, preds_close
plt.plot(trues_close[-plot_len:], label='Ground Truth (Real Price)', color='green', linewidth=2)
plt.plot(preds_close[-plot_len:], label='SDTP Prediction (JAX)', color='red', linestyle='--', linewidth=2)
plt.title(f'SDTP Improved Prediction (Last {plot_len} Days)', fontsize=14)
plt.ylabel('Price (USD)', fontsize=12)
plt.xlabel('Days', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Training Loss 曲線
plt.figure(figsize=(8, 5))
plt.plot(loss_history, linewidth=2)
plt.title('Training Loss (MSE)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# 7. 推論速度測試
# ==========================================

print("⚡ 測試推論速度...")
x_bench = jnp.array(X_test[:BATCH_SIZE])

# Warm-up
_ = forward_jit(params, x_bench, KERNEL_SIZE).block_until_ready()

# 測試
start = time.time()
n_iterations = 1000
for _ in range(n_iterations):
    _ = forward_jit(params, x_bench, KERNEL_SIZE).block_until_ready()
t_infer = time.time() - start

print(f"1000 次推論時間: {t_infer:.4f}s")
print(f"平均每次: {t_infer/n_iterations*1000:.2f}ms")
print(f"每秒可處理: {BATCH_SIZE * n_iterations / t_infer:.0f} 樣本")
```
