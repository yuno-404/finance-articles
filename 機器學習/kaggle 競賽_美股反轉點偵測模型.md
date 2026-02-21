
---
## Detecting Reversal Points in US Equities
{%preview https://www.kaggle.com/competitions/detecting-reversal-points-in-us-equities/overview %}

* 任務：金融時間序列分類 (Financial Time Series Classification)。
* 目標：找出股價走勢的「反轉點」(Swing Points)。
* 分類標籤 (3類)：
    * H (High)：局部高點 (原本的 Higher High, Lower High)。
    * L (Low)：局部低點 (原本的 Higher Low, Lower Low)。
    * None：非反轉點 (原本的 trend 或 noise)。
* 評分標準：Macro F1-Score (這意味著你不能只預測準確率高的 None，必須精準抓出稀有的 H 和 L 才能高分)。

----
### 資料
1. 極端的特徵/樣本比 (The $P \gg N$ Problem)
    * 這是這題最致命的陷阱。
        * 樣本數 ($N$)：僅 1,932 筆訓練資料。
        * 特徵數 ($P$)：高達 68,504 個特徵。
    * 解讀：平均每一筆資料有 35 個特徵在描述它。
        * 在統計學上，當特徵數遠大於樣本數時，模型極容易找到「偽相關性」(Spurious Correlation)。
        * 比喻：就像你要預測明天的股價，你只有過去 5 天的資料，但你卻蒐集了全世界 6 萬種不同的天氣、氣溫、濕度數據。你一定找得到某個數據（例如「南極的風速」）剛好跟這 5 天的股價完全吻合，但這對預測明天毫無幫助。
    * 結論：不做特徵篩選 (Feature Selection)，模型必死無疑。


3. 嚴重的類別不平衡 (Extreme Imbalance)
    * H (高點)：3.0%
    * L (低點)：2.9%
    * None (無訊號)：94.1%
    * 解讀：這是一個「大海撈針」的遊戲。
        * 如果你寫一個程式 return "None"，你的準確率 (Accuracy) 直接就是 94.1%。
        * 但這種模型在交易上價值為零，因為它抓不到任何買賣點。
    * 陷阱：題目說明寫「Evaluation Metric: Classification Accuracy」。如果這是真的，那只要全猜 None 分數就很高。但通常這種競賽的隱含目標是抓出 H/L，或者在 Tie-breaker 時會看 F1-score。


3. 數據結構與時間序 (Time Series Structure)
    * Ticker (代號)：共有 6 檔不同的金融商品 (001~006)。
    * 時間跨度：2023-04-03 到 2025-01-31。
    * 解讀：
        * 這是時間序列數據，不能隨機打亂 (Shuffle) 做驗證。
        * 不同 Ticker 之間可能存在連動性（Cross-ticker patterns），也可能有各自獨特的慣性。

----
### 整體架構
* 降維打擊：從 68,000 個特徵中，只選出最重要的 150 個。這是為了防止模型在雜訊中「幻覺」(Overfitting)。
* 時序防火牆：使用 CPCV (Combinatorial Purged Group K-Fold) 驗證，嚴格隔離訓練集與測試集，防止模型偷看未來。
* 權重槓桿：利用 Class Weights 強迫模型重視那 3% 的反轉點，否則模型只會全部預測為「無訊號」。

#### 1. 組合式淨化分組交叉驗證 (Combinatorial Purged Group K-Fold, CPCV)
* 問題：一般的 KFold 會隨機打亂數據。如果你把第 100 天的數據當測試集，第 101 天的數據當訓練集，模型會利用第 101 天的資訊「回推」第 100 天，這在現實交易中是不可能的（數據洩漏）。
* 解法 (CPCV)：
    * Group (分組)：以 date_group 為單位，確保同一天的所有股票數據都在同一邊（要嘛都在訓練集，要嘛都在測試集）。
    * Purge (淨化)：在測試集的前後，強制挖掉一段時間（PURGE_GAP = 10 天）。這是為了切斷「序列相關性」。例如，今天的波動率通常跟昨天很像，如果不隔開，模型會作弊。
    * Embargo (禁運)：在測試集之後額外挖掉一段數據（EMBARGO_PCT = 0.01）。這是為了防止長期的標籤洩漏。
    * Combinatorial (組合式)：不同於傳統 TimeSeriesSplit 只能用「過去預測未來」（導致訓練資料很少），CPCV 允許測試集在中間，訓練集在「過去 + 未來」（只要中間有 Purge 隔開）。這能最大化訓練數據的使用率。

#### 2. 兩階段特徵篩選 (Two-Stage Feature Selection)
* 原理：面對 $P \gg N$ (特徵數遠大於樣本數) 的問題，直接丟進大模型必死無疑。
* 實作 (perform_feature_selection)：
    * Lightweight Proxy Model：先用一個「輕量級」的 LightGBM (100 棵樹, 深度 3) 快速跑一遍。
    * Gain Importance：計算每個特徵的 Gain (該特徵在分裂節點時，讓 Loss 下降了多少)。Gain 比 Split 次數更能反映特徵的真實預測力。
    * Top-K：直接砍掉 Gain 排名 150 名以外的所有特徵。
* 結果解讀：日誌顯示前幾名特徵是 ratio (比率), momentum (動能), 以及一些 occurs_within_zone (價格區間訊號)。這符合金融邏輯：動能與相對價值是反轉的關鍵。

#### 3. 對抗類別不平衡 (Handling Class Imbalance)
* 現狀：None 類別佔了 94%，H 和 L 各只有 3%。
* 計算權重：
```
weights = len(y) / (3 * counts)
```
這是一種「逆頻率加權」(Inverse Frequency)。
* 結果：H 的權重是 32.91，L 是 35.19，而 None 是 1.00。
* 意義：這告訴模型：「預測錯一個 H (高點) 的懲罰，等同於預測錯 33 個 None」。這強迫模型去「賭」那些反轉點，而不是保守地全猜 None。

### 數據解讀
#### 1. 訓練過程 (The Training Loop)
* Early Stopping：大多數 Fold 在 1600~1800 棵樹左右停止，這是一個健康的收斂過程。如果 50 棵樹就停，代表學不到東西；如果 5000 棵樹還沒停，代表在過擬合。
* 分數波動：
    * Fold 1 F1: 0.3753
    * Fold 3 F1: 0.3451
    * Fold 9 F1: 0.4172

解讀：F1 分數在不同時間段（Fold）波動不大，這是一個好現象。代表模型是穩健的 (Robust)，沒有因為某些特定時間段的極端行情而失效。


#### 2. 混淆矩陣 (Confusion Matrix)
[[   4    1   72]   <- 真實是 H (77個)
 [   0    3   69]   <- 真實是 L (72個)
 [  19   17 2498]]  <- 真實是 None (2534個)

*  H (高點) 的表現：
    * 真實有 77 個 H。
    * 模型正確抓到了 4 個 (True Positive)。
    * 模型誤判成 L 有 1 個。
    * 模型漏判了 72 個 (把它們當成 None)。
* None 的表現：
    * 模型把 2498 個 None 正確預測出來。
    * 模型把 36 個 None 誤判為 H 或 L (False Positive)。

#### 3. 最終提交 (The Submission)
Distribution: {'None': 1133, 'H': 9, 'L': 9}

----

### 學學第六名
{%preview https://www.kaggle.com/competitions/detecting-reversal-points-in-us-equities/writeups/stacked-lightgbm-ensemble-for-detecting-market-rev %}

#### 用了 三個不同個性的模型
* Model 1 (Base): 標準 GBDT。
* Model 2 (Tuned): 深度正則化 (L1/L2)，防止過擬合。
* Model 3 (DART): 這是一個關鍵。DART (Dropout Additive Regression Trees) 是一種會隨機「遺忘」部分樹的算法，對於這種雜訊很多的金融數據，DART 的抗噪能力遠強於普通 GBDT。

#### Stacking (堆疊法)：
* 他是將這三個模型的預測結果 (OOF Predictions) 當作新特徵，丟給第二層的 Logistic Regression 去判斷誰說得對。這能自動學會「什麼時候該聽 DART 的，什麼時候該聽 GBDT 的」。

####  Objective Function: 
* 他使用了 multiclassova (One-vs-All)。這對類別不平衡（H/L 只有 3%）特別有效，因為它會獨立訓練 "Is it H?" 和 "Is it L?" 的二元分類器，而不是混在一起算 Softmax。


----

### 背景知識
#### GBDT (Gradient Boosting Decision Tree)
GBDT 是所有現代樹模型（XGBoost, LightGBM, CatBoost）的老祖宗。
GBDT 不是一棵樹，而是一堆樹的「加法」。
* Bagging (如 Random Forest)：並行訓練很多樹，大家投票。
* Boosting (GBDT)：串行訓練。第一棵樹學不好的地方（殘差），交給第二棵樹學；第二棵樹學不好的，交給第三棵樹……

##### 擬合殘差 (Fitting the Residuals)
想像你要預測股價是 100 元：
1. Tree 1：預測 80 元。誤差（殘差）是 $100 - 80 = 20$ 元。
2. Tree 2：它的目標不是預測 100，而是預測那個 20。假設它預測 15。現在總預測是 80+15=95。剩餘誤差 5。
3. Tree 3：目標是預測那個 5。預測了 4。
4. 最終模型：Tree 1 + Tree 2 + Tree 3 = 80+15+4=99。

為什麼叫 "Gradient" (梯度)？ 在數學上，如果你用「均方誤差」(MSE) 作為損失函數，負梯度剛好就是殘差。所以「擬合殘差」本質上就是在沿著梯度的反方向下降，讓 Loss 最小化。

#### LightGBM (Light Gradient Boosting Machine)
GBDT 雖然好，但傳統實作跑太慢。微軟開發的 LightGBM 對 GBDT 做了工程上的極致優化。它快，且準。
##### 核心優化一：Histogram-based Algorithm (直方圖算法)
* 傳統方法：為了找最佳分裂點，要把所有特徵數值排序（Sorting），非常耗內存和 CPU。
* LightGBM：把連續的浮點數（例如股價 10.1, 10.2, 10.5...）裝進 255 個桶子（Bins）變成離散的直方圖。
* 優點：記憶體佔用極低，計算速度快幾十倍，且對雜訊有天然的過濾效果（因為把細微差別模糊化了）。

##### 核心優化二：Leaf-wise Growth (按葉生長) vs. Level-wise (按層生長)
這是 LightGBM 與 XGBoost（早期版本）最大的不同。
* Level-wise (XGBoost)：像切蛋糕，每一層都要切滿，比較平衡，不容易過擬合，但效率低。
* Leaf-wise (LightGBM)：強者恆強。它會找到當前 Loss 下降最多的那個葉子繼續切，不管樹平不平衡。
    * 優點：能學到更深、更複雜的模式（高分關鍵）。
    * 缺點：容易過擬合。這就是為什麼我在程式碼中設定 max_depth 很重要的原因，必須限制它不能長太深。

##### 核心優化三：GOSS (單邊梯度採樣)
LightGBM 認為：梯度大（誤差大）的樣本對學習更有幫助，梯度小（誤差小）的樣本已經學得差不多了。
* GOSS：保留所有大梯度的數據，隨機丟棄小梯度的數據。這讓它在不損失太多精度的情況下，數據量變少，速度變快。

#### DART (Dropout Additive Regression Trees)
這是你在 Stacked Model 中看到的 Model 3。它是 LightGBM 裡面的一種特殊模式。

**痛點：過度依賴 (Over-specialization)**
在標準 GBDT 中，第一棵樹通常是最強的，後面的樹只能修修補補。這導致模型非常依賴前面的樹。如果第一棵樹學到了雜訊，後面的樹就會一直錯下去。

DART 借鑑了神經網路的 Dropout 技術。

在訓練第 k 棵樹時：
1. 隨機丟棄前面已經訓練好的某些樹（暫時假裝它們不存在）。
1. 這迫使第 $k$ 棵樹必須重新學習那些被丟棄樹的知識，而不是只學殘差。
1. 歸一化：預測時，會把樹的權重平均回來。

為什麼這對金融數據有效？
金融數據充滿雜訊 (Noise)。普通 GBDT 容易死記硬背雜訊。DART 通過「隨機遺忘」，增加了模型的不確定性與魯棒性 (Robustness)。它通常能訓練出泛化能力更強的模型，雖然訓練速度最慢，但在 Stacking 中是必備的抗噪神器。
#### CPCV (Combinatorial Purged Group K-Fold)
這是金融機器學習中最複雜、也最重要的驗證方法。如果說前面是在教模型讀書，CPCV 就是確保考試不作弊。

##### A. Group (分組)
不能把同一天（或同一個交易事件）的數據拆開。例如 10:00 的數據在訓練集，10:01 的數據在測試集，模型會直接「背答案」。
* 做法：同一天的數據，必須全部在訓練集，或全部在測試集。

##### B. Purge (淨化/隔離)
金融特徵通常有「記憶性」（例如 RSI 指標是看過去 14 天）。
* 問題：如果測試集是第 100 天，訓練集包含第 99 天。第 99 天的特徵其實包含了第 100 天的部分資訊（因為滑動窗口）。這叫 Data Leakage。
* 做法：在訓練集和測試集之間，挖掉一段 GAP（例如 10 天）。這道防火牆確保兩邊完全沒關係。

##### C. Embargo (禁運)
這是一個更細節的防護。測試集之後的一小段時間，也盡量不要馬上放回訓練集，以防標籤洩漏。

##### D. Combinatorial (組合式)
傳統的 TimeSeriesSplit（Walk-Forward）有一個缺點：早期的數據被測試的次數很少，浪費了數據。CPCV 使用組合數學 $C(N, k)$ 來生成切分。

* 例子：將數據分成 5 份 (A, B, C, D, E)，選 2 份當測試。
    * Split 1: Train [A, B, C], Test [D, E] (中間要 Purge)
    * Split 2: Train [A, B, E], Test [C, D]
    * ...
* 優點：每一份數據都有機會當訓練集，也有機會當測試集，且嚴格遵守了時間隔離。這能給出最可信的 CV 分數。



----
因為也沒辦法提繳測試新的分數了，所以下面的程式碼是我修改後的，也不是我原本的，嘗試用第六名的方法。

```
import numpy as np
import pandas as pd
import lightgbm as lgb
import gc
import time
import os
from itertools import combinations
from scipy.special import comb, softmax
from sklearn.metrics import f1_score, confusion_matrix, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: GLOBAL CONFIGURATION
# =============================================================================
class CONFIG:
    TRAIN_PATH = '/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/train.csv'
    TEST_PATH = '/kaggle/input/detecting-reversal-points-in-us-equities/new_comptetition_data/test.csv'
    SUBMIT_PATH = 'submission.csv'
    
    SEED = 42
    N_FOLDS = 5
    N_TEST_SPLITS = 2
    PURGE_GAP = 10
    EMBARGO_PCT = 0.01
    
    # Feature Selection
    N_TOP_FEATURES = 150 
    
    # Class mapping
    CLASS_MAP = {'H': 0, 'L': 1, 'None': 2}
    INV_CLASS_MAP = {0: 'H', 1: 'L', 2: 'None'}

    # Model 1: Standard GBDT
    LGB_PARAMS_1 = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'max_depth': 5,
        'num_leaves': 20,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'n_jobs': -1,
        'verbosity': -1,
        'random_state': 42
    }

    # Model 2: Regularized GBDT
    LGB_PARAMS_2 = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'n_estimators': 2500,
        'learning_rate': 0.008,
        'max_depth': 6,
        'num_leaves': 31,
        'colsample_bytree': 0.5,
        'subsample': 0.6,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'n_jobs': -1,
        'verbosity': -1,
        'random_state': 43
    }

    # Model 3: DART
    LGB_PARAMS_3 = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'dart',
        'n_estimators': 1500,
        'learning_rate': 0.02,
        'max_depth': 5,
        'num_leaves': 20,
        'colsample_bytree': 0.7,
        'xgboost_dart_mode': True,
        'uniform_drop': True,
        'drop_rate': 0.1,
        'skip_drop': 0.5,
        'n_jobs': -1,
        'verbosity': -1,
        'random_state': 44
    }

# =============================================================================
# SECTION 2: UTILITIES
# =============================================================================
def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df

class CombinatorialPurgedGroupKFold:
    def __init__(self, n_splits=5, n_test_splits=2, purge=10, pctEmbargo=0.01):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge = purge
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_groups = len(unique_groups)
        group_test_size = n_groups // self.n_splits
        split_dict = {}
        for split in range(self.n_splits):
            if split == self.n_splits - 1:
                split_dict[split] = unique_groups[int(split * group_test_size):].tolist()
            else:
                start = int(split * group_test_size)
                end = int((split + 1) * group_test_size)
                split_dict[split] = unique_groups[start:end].tolist()
        group_dict = {}
        for idx in range(len(X)):
            g = groups[idx]
            if g not in group_dict: group_dict[g] = []
            group_dict[g].append(idx)
        for test_splits in combinations(range(self.n_splits), self.n_test_splits):
            test_groups = []
            banned_groups = []
            for split in test_splits:
                current_test_group = split_dict[split]
                test_groups.extend(current_test_group)
                t_start = current_test_group[0]
                t_start_idx = np.where(unique_groups == t_start)[0][0]
                purge_start = max(0, t_start_idx - self.purge)
                banned_groups.extend(unique_groups[purge_start:t_start_idx].tolist())
                t_end = current_test_group[-1]
                t_end_idx = np.where(unique_groups == t_end)[0][0]
                embargo_size = int(n_groups * self.pctEmbargo)
                embargo_end = min(n_groups, t_end_idx + 1 + self.purge + embargo_size)
                banned_groups.extend(unique_groups[t_end_idx+1:embargo_end].tolist())
            train_groups = [g for g in unique_groups if g not in banned_groups and g not in test_groups]
            train_idx = []
            test_idx = []
            for g in train_groups: train_idx.extend(group_dict[g])
            for g in test_groups: test_idx.extend(group_dict[g])
            yield np.array(train_idx), np.array(test_idx)

# =============================================================================
# SECTION 3: FEATURE SELECTION
# =============================================================================
def perform_feature_selection(df, features, target_col, n_select=150):
    print(f"\n[Feature Selection] Processing {len(features)} features...")
    selector_df = df[features].copy()
    selector_df = reduce_mem_usage(selector_df, verbose=False)
    
    lgb_params = {
        'objective': 'multiclass', 'num_class': 3, 'metric': 'multi_logloss',
        'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1,
        'random_state': 42, 'verbosity': -1, 'n_jobs': -1
    }
    
    y = df[target_col]
    counts = np.bincount(y)
    weights = len(y) / (3 * counts)
    sample_weights = [weights[label] for label in y]
    
    dtrain = lgb.Dataset(selector_df, label=y, weight=sample_weights)
    model = lgb.train(lgb_params, dtrain)
    
    importance = model.feature_importance(importance_type='gain')
    feature_imp = pd.DataFrame({'feature': features, 'importance': importance})
    feature_imp = feature_imp.sort_values('importance', ascending=False)
    
    top_features = feature_imp.head(n_select)['feature'].tolist()
    print(f"  Selected {len(top_features)} features.")
    del selector_df, dtrain, model
    gc.collect()
    return top_features

# =============================================================================
# SECTION 4: TRAINING & STACKING
# =============================================================================
def train_stacking_ensemble(train_df, test_df, features):
    print("\n" + "=" * 80)
    print("SECTION: STACKING ENSEMBLE TRAINING")
    print("=" * 80)
    
    X = train_df[features].values
    y = train_df['target'].values
    groups = train_df['date_group'].values
    X_test = test_df[features].values
    
    # Class Weights
    counts = np.bincount(y)
    weights = len(y) / (3 * counts)
    weights = weights / weights[2]
    print(f"Weights: H={weights[0]:.2f}, L={weights[1]:.2f}, None={weights[2]:.2f}")
    
    # CV Strategy
    cv = CombinatorialPurgedGroupKFold(
        n_splits=CONFIG.N_FOLDS, n_test_splits=CONFIG.N_TEST_SPLITS,
        purge=CONFIG.PURGE_GAP, pctEmbargo=CONFIG.EMBARGO_PCT
    )
    
    # 計算實際折數
    n_actual_folds = int(comb(CONFIG.N_FOLDS, CONFIG.N_TEST_SPLITS))
    print(f"Total CV Folds: {n_actual_folds}")
    
    # Base Models Configuration
    model_configs = [
        ("M1_GBDT", CONFIG.LGB_PARAMS_1),
        ("M2_Reg", CONFIG.LGB_PARAMS_2),
        ("M3_DART", CONFIG.LGB_PARAMS_3)
    ]
    
    # Meta-Features Storage
    oof_meta_features = np.zeros((len(train_df), 9))
    test_meta_features = np.zeros((len(test_df), 9))
    
    # === 修正：為每個模型重新創建生成器 ===
    for model_idx, (model_name, params) in enumerate(model_configs):
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        print(f"{'='*60}")
        
        current_oof_preds = np.zeros((len(train_df), 3))
        current_oof_counts = np.zeros(len(train_df))
        current_test_preds = np.zeros((len(test_df), 3))
        fold_scores = []
        
        # 重新生成折疊
        fold_num = 0
        for train_idx, val_idx in cv.split(X, y, groups=groups):
            fold_num += 1
            print(f"  Fold {fold_num}/{n_actual_folds} - Train: {len(train_idx)}, Val: {len(val_idx)}")
            
            w_train = np.array([weights[label] for label in y[train_idx]])
            
            dtrain = lgb.Dataset(X[train_idx], label=y[train_idx], weight=w_train)
            dval = lgb.Dataset(X[val_idx], label=y[val_idx], reference=dtrain)
            
            # Callbacks
            cbs = [lgb.log_evaluation(period=0)]
            if params.get('boosting_type') != 'dart':
                cbs.append(lgb.early_stopping(stopping_rounds=50, verbose=False))
            
            model = lgb.train(params, dtrain, valid_sets=[dval], callbacks=cbs)
            
            # Predict
            val_probs = softmax(model.predict(X[val_idx]), axis=1)
            test_probs = softmax(model.predict(X_test), axis=1)
            
            current_oof_preds[val_idx] += val_probs
            current_oof_counts[val_idx] += 1
            current_test_preds += test_probs
            
            # Score
            val_pred_lbl = np.argmax(val_probs, axis=1)
            f1 = f1_score(y[val_idx], val_pred_lbl, average='macro')
            fold_scores.append(f1)
            print(f"    Macro F1: {f1:.4f}")
            
            del model, dtrain, dval
            gc.collect()
        
        # Average predictions
        current_oof_preds /= np.maximum(current_oof_counts[:, np.newaxis], 1)
        current_test_preds /= fold_num  # 使用實際折數
        
        print(f"\n  >>> {model_name} Average Macro F1: {np.mean(fold_scores):.4f} ±{np.std(fold_scores):.4f}")
        
        # Store meta-features
        start_col = model_idx * 3
        oof_meta_features[:, start_col:start_col+3] = current_oof_preds
        test_meta_features[:, start_col:start_col+3] = current_test_preds

    # === LEVEL 1: Meta-Learner ===
    print("\n" + "=" * 80)
    print("TRAINING META-LEARNER (Logistic Regression)")
    print("=" * 80)
    
    scaler = StandardScaler()
    X_meta_train = scaler.fit_transform(oof_meta_features)
    X_meta_test = scaler.transform(test_meta_features)
    
    sample_weights_meta = np.array([weights[label] for label in y])
    
    meta_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        C=1.0, 
        random_state=42,
        max_iter=1000
    )
    
    meta_model.fit(X_meta_train, y, sample_weight=sample_weights_meta)
    
    # Final Predictions
    final_test_probs = meta_model.predict_proba(X_meta_test)
    final_test_labels = np.argmax(final_test_probs, axis=1)
    
    # OOF Performance
    oof_meta_pred = meta_model.predict(X_meta_train)
    oof_score = f1_score(y, oof_meta_pred, average='macro')
    mcc_score = matthews_corrcoef(y, oof_meta_pred)
    
    print(f"\n{'='*80}")
    print(">>> FINAL STACKING RESULTS <<<")
    print(f"{'='*80}")
    print(f"Meta-Model OOF Macro F1: {oof_score:.4f}")
    print(f"Meta-Model OOF MCC:      {mcc_score:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, oof_meta_pred))
    print(f"{'='*80}\n")
    
    return final_test_labels

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    start_time = time.time()
    
    print("=" * 80)
    print("REVERSAL POINT DETECTION - STACKING ENSEMBLE")
    print("=" * 80)
    
    # 1. Load Data
    print("\n[1/6] Loading Data...")
    train_df = pd.read_csv(CONFIG.TRAIN_PATH)
    test_df = pd.read_csv(CONFIG.TEST_PATH)
    print(f"  Train: {train_df.shape}, Test: {test_df.shape}")
    
    # 2. Preprocess
    print("\n[2/6] Preprocessing...")
    train_df['class_label'] = train_df['class_label'].fillna('None')
    train_df['target'] = train_df['class_label'].map(CONFIG.CLASS_MAP).fillna(2).astype(int)
    train_df['date_group'] = pd.to_datetime(train_df['t']).apply(lambda x: x.toordinal())
    
    print(f"  Target Distribution: {dict(train_df['target'].value_counts().sort_index())}")
    
    # 3. Feature Selection
    print("\n[3/6] Selecting Features...")
    meta_cols = ['id', 'ticker_id', 't', 'class_label', 'target', 'date_group', 'Unnamed: 0']
    features = sorted(list((set(train_df.columns) - set(meta_cols)) & set(test_df.columns)))
    
    selected_features = perform_feature_selection(
        train_df, features, 'target', n_select=CONFIG.N_TOP_FEATURES
    )
    
    # 4. Memory Optimization
    print("\n[4/6] Optimizing Memory...")
    train_df[selected_features] = reduce_mem_usage(train_df[selected_features])
    test_df[selected_features] = reduce_mem_usage(test_df[selected_features])
    
    # 5. Train Stacking Ensemble
    print("\n[5/6] Training Stacking Ensemble...")
    test_labels = train_stacking_ensemble(train_df, test_df, selected_features)
    
    # 6. Generate Submission
    print("\n[6/6] Generating Submission...")
    submission = pd.DataFrame({
        'id': test_df['id'],
        'class_label': [CONFIG.INV_CLASS_MAP[label] for label in test_labels]
    })
    submission.to_csv(CONFIG.SUBMIT_PATH, index=False)
    
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"✓ Submission saved to: {CONFIG.SUBMIT_PATH}")
    print(f"✓ Total Time: {elapsed/60:.2f} minutes")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
```
