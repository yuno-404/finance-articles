---
title: 如何收割波動性？
tags: [Trading]

---

## 算術平均 vs. 幾何平均的陷阱
首先，我們要打破一個直覺。假設你有 100 元，投資一個資產。第一年漲了 50%（變成 150 元）。第二年跌了 50%（變成 75 元）。
直覺告訴你：漲 50% 跌 50%，平均起來應該是不賺不賠吧？錯！你虧了 25 元（-25%）。這就是金融學著名的「波動性拖累」（Volatility Drag）。
* 算術平均（Arithmetic Mean）： $(+50\% - 50\%) / 2 = 0\%$。這是你以為的期望值。$$R_A = \frac{1}{N} \sum_{i=1}^{N} r_i$$
* 幾何平均（Geometric Mean）： 這是你實際口袋裡的錢，它永遠小於或等於算術平均。

$$R_G = \left( \prod_{i=1}^{N} (1+r_i) \right)^{1/N} - 1$$
$$R_G \approx R_A - \frac{\sigma^2}{2}$$
波動越大，這兩者之間的差距（拖累）就越大。傳統投資者視波動為「風險」，因為它會吃掉你的複利增長。但「蛛網收割」策略者卻視波動為「資源」，他們想辦法把這個被吃掉的收益「吐」出來。

## 薛農的惡魔：如何利用「再平衡」逆天改命？
想像一支股票價格隨機亂跳，長期趨勢不明。如果你只是「買入持有」（Buy and Hold），你的資產也會跟著亂跳，甚至因為波動性拖累而縮水。

但如果你採用「50/50 再平衡策略」：
當你只有一半資金在股票裡，整個投資組合的波動率 $\sigma_p$ 會變成原來的一半：$$\sigma_p = 0.5 \times \sigma$$
將資金分為兩半：50% 持有現金，50% 持有股票。
每天（或定期）檢查：
* 若股票漲了，股票佔比會超過 50%，你就賣出多餘的股票變現（獲利了結）。
* 若股票跌了，股票佔比會低於 50%，你就動用現金買入股票（低價吸籌）。

代入到幾何收益公式
$$R_G(\text{Rebalanced}) = 0.5\mu - 0.125\sigma^2$$

即便股票價格最終跌回原點，甚至微跌，只要過程中波動夠大，你通過不斷的「高拋低吸」，帳戶總資金竟然是增長的！你收割的不是股價的漲幅，而是波動本身

## 槓桿 $f$ 的數學秘密
為什麼簡單的「維持固定比例」就能賺錢？這背後有一個精妙的數學公式控制著你的交易行為。

我們定義 $f$ 為你的目標槓桿率（Target Leverage）。
如果你全額買股，不留現金，$f=1$。如果你一半現金一半股，$f=0.5$。當價格變動率為 $r$ 時，為了維持槓桿 $f$ 不變，你需要進行的交易量 $\Delta V$ 是：$$\Delta V = \frac{(f-1)r}{1+r} \times \text{初始持倉}$$

1. 當 $f=1$（買入持有）：$(1-1) = 0$，交易量為 0。你不做任何操作，命運完全交給市場。
2. 當 $f > 1$（融資加槓桿）：$(f-1)$ 是正數。價格漲（$r>0$）$\rightarrow$ 你要買入（加倉）。價格跌（$r<0$）$\rightarrow$ 你要賣出（停損）。這是典型的「追漲殺跌」，適合大牛市，但在震盪市中會被雙巴掌打死。
3. 當 $f < 1$（蛛網收割，例如 $f=0.5$）：$(f-1)$ 是負數。價格漲（$r>0$）$\rightarrow$ 分子為負 $\rightarrow$ 你自動賣出。價格跌（$r<0$）$\rightarrow$ 分子為正 $\rightarrow$ 你自動買入。


## 策略比較
策略 A：死多頭（$f=1$）100 $\to$ 32 $\to$ 100。
* 結果：不賺不賠，浪費時間，還嚇出一身冷汗。

策略 B：固定金額投資（定投/馬丁格爾）
* 越跌越買，而且堅持每次都要買固定金額。
* 風險：在跌到 32 元的過程中，你的現金可能早就燒光了，甚至面臨爆倉風險。這是在賭命。

策略 C：波動收割（$f=0.5$ 固定槓桿）
* 下跌時： 雖然總資產縮水，但因為你需要維持 50% 倉位，你會用手中的現金不斷買入變得便宜的籌碼。
* 觸底時： 在 32 元時，你手中的股數遠多於 100 元時。
* 上漲時： 隨著價格回升，你開始慢慢賣出股票，鎖定利潤。
* 結局： 當價格回到 100 元時，你的資產總值會超過初始本金（例如變成 106 元）。那多出來的 6% 是哪來的？它不來自資產增值（因為價格沒變），它來自於你在下跌途中低價囤積的籌碼，在上漲途中高價賣出的差價。這就是「再平衡溢價」（Rebalancing Premium）

## 注意事項
1. 選對標的： 這套策略最怕「跌下去不回來」（歸零）。所以不要用在單一垃圾股或高風險小幣上。指數型 ETF（如 SPY, 0050）或一籃子資產是最佳選擇，因為它們長期不會歸零且波動性適中。
1. 資金管理是關鍵： 堅持「固定槓桿」（Fixed Leverage）而非「固定金額」。固定槓桿能確保你永遠有現金抄底，且在暴跌時不會爆倉（因為你的虧損是按比例的）。
1. 注意摩擦成本： 頻繁的買賣會產生手續費和稅。不要股價跳動 0.1% 就去再平衡。設定一個閾值（Threshold），例如當倉位偏離目標 5% 時再動手，這樣能最大化「波動性紅利」與「交易成本」之間的效益。



## 推導過程
----
槓桿公式
$$E_0 = \frac{P_0 V_0}{f}$$
* $E_0$ (Equity)：你的本金（你實際掏出的錢）。
* $P_0 V_0$ (Position Value)：房子/股票的總價值（單價 $\times$ 數量）。
* $f$ (Leverage)：槓桿倍數（你把本金放大了幾倍）。

利用「槓桿倍數 $f$」來快速判斷負債比例
$$E_0 - P_0 V_0 = P_0 V_0 (\frac{1}{f} - 1)$$
* 若 $f=2$ (2倍槓桿)：$(\frac{1}{2} - 1) = -0.5$。 $\rightarrow$ 負債是總倉位的一半。

什麼是 $E_{temp}$？
Mark-to-Market（市值計價） 的意思是：雖然你還沒賣掉資產，但如果現在立刻結算，你的帳戶有多少錢？
$$E_{temp} = E_0 + \text{Profit/Loss}$$
$$\text{Profit} = (P_1 - P_0) \times V_0$$
$P_1 = P_0(1+r)$
$$\text{Profit} = P_0 r V_0$$
$$E_{temp} = \underbrace{\frac{P_0 V_0}{f}}_{\text{本金}} + \underbrace{P_0 r V_0}_{\text{賺的錢}}$$
$$E_{temp} = P_0 V_0 \times \left( \frac{1}{f} + r \right)$$

現在的總權益，等於 「原本資產總值」去乘以「(本金比率 + 漲跌幅)」。

## 再平衡推導
$$P_1 V_1 = E_1 \times f$$
* $P_1 V_1$：新的總資產價值（新價格 $\times$ 新數量）。
* $E_1 \times f$：新的本金 $\times$ 設定的槓桿倍數。


$$P_0 (1+r) V_1 = \left[ P_0 V_0 (\frac{1}{f} + r) \right] \times f$$
* 左邊：把 $P_1$ 換成 $P_0(1+r)$。
* 右邊：把 $E_1$ 換成上一題算出來的公式 $P_0 V_0 (\frac{1}{f} + r)$。

$$V_0 (\frac{1}{f} + r) \times f$$
$$(1+r) V_1 = V_0 (1 + fr)$$
$$(1+r)(V_0 + \Delta V) = V_0 + frV_0$$
$$V_0 + \Delta V + rV_0 + r\Delta V = V_0 + frV_0$$
$$\Delta V + rV_0 + r\Delta V = frV_0$$
$$\Delta V + r\Delta V = frV_0 - rV_0$$
$$\Delta V(1+r) = (f-1)rV_0$$
$$\Delta V = \frac{(f-1)r}{1+r} V_0$$

## 程式碼回測
```
# @title 蛛網收割 (Fixed Leverage) 策略回測與最佳化引擎
# @markdown 請直接執行此區塊進行模擬。你可以修改參數來測試不同標的。

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar

# ==========================================
# 1. 設定參數 (Configuration)
# ==========================================
TICKER = "LEU" # @param {type:"string"}
START_DATE = "2025-08-01" # @param {type:"string"}
END_DATE = "2025-12-26" # @param {type:"string"}
INITIAL_CAPITAL = 100000 # @param {type:"number"}
TARGET_LEVERAGE = 0.5 # @param {type:"number"}
COST_RATE = 0.001 # @param {type:"number"}

# ==========================================
# 2. 核心策略邏輯 (Core Strategy Logic)
# ==========================================
def run_strategy(prices, target_leverage, threshold, cost_rate, initial_capital):
    """
    執行固定槓桿策略，並帶有再平衡閾值與交易成本。
    """
    n = len(prices)
    cash_arr = np.zeros(n)
    shares_arr = np.zeros(n)
    equity_arr = np.zeros(n)
    trade_count = 0
    turnover = 0.0

    # 初始化
    cash = initial_capital
    shares = 0

    # 第一筆交易：建立初始部位
    target_position = initial_capital * target_leverage
    shares = target_position / prices[0]
    transaction_val = shares * prices[0]
    cost = transaction_val * cost_rate
    cash = initial_capital - transaction_val - cost
    trade_count += 1
    
    cash_arr[0] = cash
    shares_arr[0] = shares
    equity_arr[0] = cash + shares * prices[0]

    for t in range(1, n):
        # Mark-to-Market
        current_price = prices[t]
        current_position_val = shares * current_price
        current_equity = cash + current_position_val
        
        # 計算目標部位與偏差
        target_position_val = current_equity * target_leverage
        deviation = current_position_val - target_position_val
        deviation_pct = abs(deviation) / current_equity if current_equity > 0 else 0

        # 檢查是否觸發閾值
        if deviation_pct > threshold:
            # 執行再平衡
            trade_val = -deviation
            trade_cost = abs(trade_val) * cost_rate
            
            # 更新狀態
            shares = shares + (trade_val / current_price)
            cash = cash - trade_val - trade_cost
            
            trade_count += 1
            turnover += abs(trade_val)
        
        # 記錄當前狀態
        cash_arr[t] = cash
        shares_arr[t] = shares
        equity_arr[t] = cash + shares * current_price

    return equity_arr, trade_count, turnover

# ==========================================
# 3. 績效評估指標 (Performance Metrics)
# ==========================================
def calculate_metrics(equity_curve):
    """
    計算關鍵績效指標
    """
    returns = pd.Series(equity_curve).pct_change().dropna()
    
    # CAGR
    total_days = len(equity_curve)
    years = total_days / 252
    cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1 if years > 0 else 0
    
    # Volatility (Annualized)
    vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
    
    # Sharpe Ratio
    sharpe = (cagr / vol) if vol != 0 else 0
    
    # Max Drawdown
    peak = pd.Series(equity_curve).expanding(min_periods=1).max()
    drawdown = (pd.Series(equity_curve) - peak) / peak
    max_dd = drawdown.min()
    
    # Calmar Ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = (cagr / downside_std) if downside_std != 0 else 0

    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "Calmar": calmar,
        "Sortino": sortino
    }

# ==========================================
# 4. 執行與數據獲取
# ==========================================
print(f"下載數據: {TICKER}...")
data = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)

if len(data) == 0:
    print("下載失敗，改用隨機幾何布朗運動 (GBM) 模擬...")
    np.random.seed(42)
    days = 1000
    mu = 0.1
    sigma = 0.5
    dt = 1/252
    prices = 100 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal(days)))
    dates = pd.date_range(start=START_DATE, periods=days)
    data = pd.DataFrame({"Close": prices}, index=dates)
    prices = data['Close'].values
else:
    # 處理 yfinance 多層索引
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close'].iloc[:, 0].values
    else:
        prices = data['Close'].values

# 基準：買入持有
bnh_equity = (prices / prices[0]) * INITIAL_CAPITAL

# ==========================================
# 5. 最佳化閾值 (Optimization Loop)
# ==========================================
print("開始最佳化閾值 (Grid Search)...")
thresholds = np.linspace(0.0, 0.15, 30)
results = []

for th in thresholds:
    eq, count, turn = run_strategy(prices, TARGET_LEVERAGE, th, COST_RATE, INITIAL_CAPITAL)
    mets = calculate_metrics(eq)
    mets['Threshold'] = th
    mets['Trade_Count'] = count
    mets['Final_Equity'] = eq[-1]
    results.append(mets)

results_df = pd.DataFrame(results)

# 找出最佳閾值
best_result = results_df.loc[results_df['Sharpe'].idxmax()]
best_threshold = best_result['Threshold']

# 使用最佳閾值再跑一次
final_equity, final_count, final_turn = run_strategy(prices, TARGET_LEVERAGE, best_threshold, COST_RATE, INITIAL_CAPITAL)
final_metrics = calculate_metrics(final_equity)

# ==========================================
# 6. 視覺化與報告
# ==========================================
plt.figure(figsize=(14, 10))
plt.style.use('seaborn-v0_8-darkgrid')

# 圖 1: 權益曲線比較
plt.subplot(2, 2, 1)
plt.plot(data.index, bnh_equity, label='Buy & Hold (100%)', color='gray', alpha=0.6, linewidth=2)
plt.plot(data.index, final_equity, label=f'Spiderweb (Lev={TARGET_LEVERAGE}, Th={best_threshold:.1%})', color='blue', linewidth=2)
plt.title(f'Strategy Comparison: {TICKER}', fontsize=14, fontweight='bold')
plt.ylabel('Equity ($)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# 圖 2: 閾值對績效的影響
plt.subplot(2, 2, 2)
plt.plot(results_df['Threshold']*100, results_df['Sharpe'], marker='o', color='green', linewidth=2)
plt.axvline(best_threshold*100, color='red', linestyle='--', alpha=0.7, label=f'Best: {best_threshold:.1%}')
plt.title('Threshold vs Sharpe Ratio', fontsize=14, fontweight='bold')
plt.xlabel('Rebalance Threshold (%)', fontsize=12)
plt.ylabel('Sharpe Ratio', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)

# 圖 3: 閾值對交易次數的影響
plt.subplot(2, 2, 3)
plt.plot(results_df['Threshold']*100, results_df['Trade_Count'], marker='x', color='red', linewidth=2)
plt.title('Threshold vs Trade Count', fontsize=14, fontweight='bold')
plt.xlabel('Rebalance Threshold (%)', fontsize=12)
plt.ylabel('Number of Trades', fontsize=12)
plt.grid(True)

# 圖 4: 回撤比較
plt.subplot(2, 2, 4)
bnh_peak = pd.Series(bnh_equity).expanding(min_periods=1).max()
bnh_dd = (pd.Series(bnh_equity) - bnh_peak) / bnh_peak
strategy_peak = pd.Series(final_equity).expanding(min_periods=1).max()
strategy_dd = (pd.Series(final_equity) - strategy_peak) / strategy_peak
plt.plot(data.index, bnh_dd*100, label='B&H Drawdown', color='gray', alpha=0.6)
plt.plot(data.index, strategy_dd*100, label='Strategy Drawdown', color='blue', linewidth=2)
plt.title('Drawdown Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Drawdown (%)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 文字報告
print("\n" + "="*50)
print(f"最佳化結果分析 ({TICKER})")
print("="*50)
print(f"測試期間: {START_DATE} 至 {END_DATE}")
print(f"初始資金: ${INITIAL_CAPITAL:,.0f}")
print(f"目標槓桿: {TARGET_LEVERAGE:.1%}")
print(f"交易成本: {COST_RATE:.2%}")
print("-" * 50)
print(f"最佳再平衡閾值: {best_threshold:.2%}")
print(f"策略年化報酬 (CAGR): {final_metrics['CAGR']:.2%}")
print(f"策略波動率 (Vol): {final_metrics['Volatility']:.2%}")
print(f"策略夏普值 (Sharpe): {final_metrics['Sharpe']:.2f}")
print(f"策略索提諾比率 (Sortino): {final_metrics['Sortino']:.2f}")
print(f"策略最大回撤 (MaxDD): {final_metrics['MaxDD']:.2%}")
print(f"策略卡瑪比率 (Calmar): {final_metrics['Calmar']:.2f}")
print(f"總交易次數: {final_count}")
print(f"期末資金: ${final_equity[-1]:,.0f}")
print("-" * 50)
bnh_metrics = calculate_metrics(bnh_equity)
print(f"買入持有 CAGR: {bnh_metrics['CAGR']:.2%}")
print(f"買入持有 Sharpe: {bnh_metrics['Sharpe']:.2f}")
print(f"買入持有 MaxDD: {bnh_metrics['MaxDD']:.2%}")
print(f"買入持有期末資金: ${bnh_equity[-1]:,.0f}")
print("="*50)

# 績效摘要表
print("\n閾值最佳化結果 (前5名):")
print(results_df.nlargest(5, 'Sharpe')[['Threshold', 'CAGR', 'Sharpe', 'MaxDD', 'Trade_Count']].to_string(index=False))
```

