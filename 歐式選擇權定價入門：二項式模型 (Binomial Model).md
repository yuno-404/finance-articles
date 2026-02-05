---
title: 歐式選擇權定價入門：二項式模型 (Binomial Model)

---

## 選擇權定價入門：二項式模型 (Binomial Model)
內容來源於: 金融巴別塔
進行整理筆記，因為是完全新手導向，會有基本介紹。
推薦去看影片，講得很清楚。

----
## 基礎介紹
* Call Option (歐式買權)：讓你在到期日那天，有權利以「約定好的價格」買入股票。如果股價漲了，你就賺了；如果股價跌了，你大不了不買（價值歸零）。
* Put Option (歐式賣權)：讓你在到期日那天，有權利以「約定好的價格」賣出股票。
* Arbitrage Opportunity (套利機會)：這在金融學裡是大忌。意思是不承擔任何風險，卻能獲得比「無風險利率」還高的利潤。所有的定價模型都是建立在「無套利機會」(No Arbitrage) 的假設上——也就是說，市場是效率的，沒有白吃的午餐。
* Risk Neutral (風險中性)：這是一個數學上的假設狀態，在這個狀態下，投資人不在乎風險，只在乎期望報酬。這會導出我們後面最強大的公式。

----

我們用最簡單的「一步二項樹」來舉例。
* 標的物：股票 ($S$)現在股價 ($S_0$)：20 元
* 時間長度 ($t$)：3 個月 (0.25 年)
* 未來股價變動：
	* 上漲 ($u$, up)：變成 22 元 (上漲因子 $u = 22/20 = 1.1$)
	* 下跌 ($d$, down)：變成 18 元 (下跌因子 $d = 18/20 = 0.9$)
* 無風險利率 ($r$)：12% (年利率)
* 買權履約價 ($K$)：21 元

#### 步驟 1：計算到期日的選擇權價值
3個月後，如果我有這個買權 (Strike = 21)：
* 情況 A (股價漲到 22)：我可以花 21 元買市價 22 元的股票，賺了 1 元。
	* $f_u = 1$
* 情況 B (股價跌到 18)：市價 18 比履約價 21 還便宜，我直接去市場買就好，不會執行選擇權。
	* $f_d = 0$

**問題來了：那這個選擇權「現在」應該賣多少錢 ($f$)？**
#### 步驟 2：建構無風險投資組合 (Delta Hedging)
券商賣給你一張選擇權，他承擔了風險。為了消除這個風險，券商會去買入一定數量的股票 ($\Delta$) 來對沖。
我們要設計一個投資組合：「買入 $\Delta$ 股股票 + 賣出 1 單位選擇權」。目標是：無論股價漲或跌，這個組合的價值都一樣（無風險）。

* 若股價上漲 (Up State)：
	* 手上的股票價值：$\Delta \times 22$
	* 賣出的選擇權賠錢 (要付給客戶)：$-1$
	* 總價值：$22\Delta - 1$
* 若股價下跌 (Down State)：
	* 手上的股票價值：$\Delta \times 18$
	* 賣出的選擇權沒事 (歸零)：$-0$
	* 總價值：$18\Delta$
#### 步驟 3：解方程式求 $\Delta$
因為要「無風險」，所以兩種情況的價值必須相等：$$22\Delta - 1 = 18\Delta$$
移項運算：
$$4\Delta = 1$$
$$\Delta = 0.25$$

**意義：券商每賣出一單位選擇權，就要買入 0.25 股的股票來避險。**
#### 步驟 4：計算現值
既然確定了 $\Delta = 0.25$，我們把它代回去算未來的組合價值：
未來價值 = $18 \times 0.25 = 4.5$ (驗算上漲邊：$22 \times 0.25 - 1 = 4.5$，正確！)
這個組合 3 個月後價值確定是 4.5 元。因為它是「無風險」的，所以它的報酬率應該等於「無風險利率 (12%)」。我們要把它折現 (Discount) 回今天：$$現在價值 = 4.5 \times e^{-0.12 \times 0.25}$$
(註：$e$ 是自然指數，用來算連續複利折現，$0.25$ 是時間年份)$$現在價值 \approx 4.367$$

#### 步驟 5：回推選擇權價格 ($f$)
我們知道投資組合今天的組成是：「買 0.25 股股票」扣掉「賣出選擇權的價值 ($f$)」。根據 一價法則 (Law of One Price)，組合的成本必須等於剛剛算出的現值：$$0.25 \times 20 (\text{股價}) - f = 4.367$$
$$5 - f = 4.367$$$$f = 0.633$$
答案算出！這個買權合理的價格是 0.633 元。

有沒有更快的算法?

----
## 風險中性評價法 (Risk Neutral Valuation)
如果每次都要解聯立方程式太慢了。透過上面的邏輯，數學家整理出一個「通用公式」，可以秒殺這個問題。
#### 1. 什麼是風險中性機率 ($p$)？
這是一個假想的機率，在這個機率下，股票的期望報酬率剛好等於無風險利率。公式如下：$$p = \frac{e^{rt} - d}{u - d}$$
* $e^{rt}$：無風險利率成長因子
* $d$：下跌因子
* $u$：上漲因子

#### 2. 計算範例中的 $p$
我們把數字代進去：
* $r = 0.12$, $t = 0.25$, $u = 1.1$, $d = 0.9$
* $e^{0.12 \times 0.25} = e^{0.03} \approx 1.03045$

$$p = \frac{1.03045 - 0.9}{1.1 - 0.9} = \frac{0.13045}{0.2} = 0.6523$$
這代表在數學模型中，上漲的「機率」被視為 65.23%。
#### 3. 套用超強公式算價格
有了 $p$，選擇權價格 ($f$) 就是把未來的期望值算出來，再折現回來：$$f = e^{-rt} [ p \times f_u + (1-p) \times f_d ]$$

代入數字：$$f = e^{-0.03} [ 0.6523 \times 1 + (1 - 0.6523) \times 0 ]$$$$f = 0.9704 \times [ 0.6523 ]$$$$f \approx 0.633$$

算出來的答案一模一樣，但是速度快非常多！ 這個公式最神奇的地方在於：它完全不需要知道真實世界中股票上漲的機率，也不需要知道投資人的風險偏好。
這是最大的重點，**無關漲跌和偏好**!

----
## 兩步二項樹 (2-Step Binomial Tree)
現實生活中，時間是連續的，我們可以用更多步驟來模擬。現在我們把時間拉長，或者切得更細，變成兩步。
#### 1. 範例情境 (賣權 Put Option)
目前股價 ($S_0$)：50 元 (注意：這裡換個例子，用影片後段的賣權範例)
履約價 ($K$)：52 元
上漲/下跌因子：假設 $u=1.2$, $d=0.8$
步驟：兩步 (2 Steps)

#### 2. 倒推法 (Backward Induction)
要算今天的價格，我們要從「最未來」往回推。
Step A: 算出最後節點 (T=2) 的選擇權價值
* 股價路徑：$50 \to 40 \to 32$ (連跌兩次)
* 此時股價 32，履約價 52。
* 因為是賣權 (Put)，我有權用 52 賣掉市價 32 的股票。
* 獲利 = $52 - 32 = 20$。

Step B: 算出中間節點 (T=1) 的價值假設我們已經算出 $p$ (風險中性機率) 是 0.6282。我們站在 $S=40$ (跌一次) 的這個節點看未來：
* 如果再漲 (變成 48)：賣權價值 = $52-48 = 4$
* 如果再跌 (變成 32)：賣權價值 = 20
* T=1 的選擇權價值 = $e^{-rt} [ 0.6282 \times 4 + (1-0.6282) \times 20 ]$
* 算出結果後，填入中間節點。

Step C: 算出現在 (T=0) 的價值 利用剛剛算出的 T=1 的兩個節點價值，再次套用超強公式，折現回今天。
這就是所謂的「倒推法」，先算終點，一步步推回起點

當然也是可以用公式解

```
import math

def detailed_binomial_pricing():
    print("="*60)
    print("      二項式模型完整計算過程 (Step-by-Step vs Formula)")
    print("="*60)

    # --- 1. 設定題目參數 (依照圖片數據) ---
    S0 = 50.0       # 現在股價
    K = 52.0        # 履約價
    r = 0.05        # 無風險利率 (5%)
    u = 1.2         # 上漲因子
    d = 0.8         # 下跌因子
    T_step = 1.0    # 每一個步驟的時間長度 (年)
    N = 2           # 總步數 (2 Steps)

    # --- 2. 計算關鍵參數 ---
    # 折現因子 (Discount Factor) = e^(-rt)
    df = math.exp(-r * T_step)
    
    # 風險中性機率 p
    # 公式: p = (e^(rt) - d) / (u - d)
    p = (math.exp(r * T_step) - d) / (u - d)
    q = 1 - p  # 下跌機率

    print(f"【基礎參數】")
    print(f"  初始股價 (S0): {S0}")
    print(f"  上漲因子 (u):  {u}")
    print(f"  下跌因子 (d):  {d}")
    print(f"  單步折現因子:  {df:.4f}")
    print(f"  上漲機率 (p):  {p:.4f}")
    print(f"  下跌機率 (q):  {q:.4f}")
    print("-" * 60)

    # ==========================================
    # 方法一：逐步倒推法 (Backward Induction)
    # ==========================================
    print("\n【方法一：逐步倒推法 (畫圖邏輯)】")
    
    # 步驟 A: 算出 T=2 (到期日) 的所有可能股價與 Payoff
    # 節點順序：[跌跌, 漲跌, 漲漲] -> [DD, UD, UU]
    # 股價分別是: 32, 48, 72
    
    # 建立一個 list 來存儲每一層的選擇權價值
    option_values = []
    stock_prices_T2 = []
    
    print(f"  [T=2 到期日]：")
    for i in range(N + 1):
        # i 是上漲次數，N-i 是下跌次數
        S_final = S0 * (u ** i) * (d ** (N - i))
        stock_prices_T2.append(S_final)
        
        # 賣權 Payoff = max(K - S, 0)
        payoff = max(K - S_final, 0)
        option_values.append(payoff)
        
        state = "UU" if i==2 else ("UD" if i==1 else "DD")
        print(f"    節點 {state} (漲{i}次): 股價={S_final:.1f}, Payoff={payoff:.1f}")

    print(f"    -> T=2 價值陣列: {option_values}")

    # 步驟 B: 開始倒推回 T=1
    print(f"\n  [T=1 中間過程] (倒推計算)：")
    # 我們有 3 個終點值，會縮減成 2 個中間值
    new_values = []
    for i in range(N):
        f_down = option_values[i]   # 下一格
        f_up = option_values[i+1]   # 上一格
        
        # 套用單步公式
        val = df * (p * f_up + q * f_down)
        new_values.append(val)
        
        state = "U" if i==1 else "D"
        print(f"    節點 {state}: {df:.4f} × ({p:.4f}×{f_up:.1f} + {q:.4f}×{f_down:.1f}) = {val:.4f}")
    
    option_values = new_values
    print(f"    -> T=1 價值陣列: {[round(v, 4) for v in option_values]}")

    # 步驟 C: 倒推回 T=0 (現在)
    print(f"\n  [T=0 現在] (最終倒推)：")
    final_value_tree = 0
    # 這時只剩下兩個值，縮減成 1 個
    f_down = option_values[0]
    f_up = option_values[1]
    
    final_value_tree = df * (p * f_up + q * f_down)
    print(f"    節點 A: {df:.4f} × ({p:.4f}×{f_up:.4f} + {q:.4f}×{f_down:.4f})")
    print(f"    -> ★ T=0 最終價值: {final_value_tree:.4f}")

    # ==========================================
    # 方法二：超強公式法 (Direct Formula)
    # ==========================================
    print("-" * 60)
    print("\n【方法二：超強公式法 (一次到位)】")
    print("  說明：直接將所有終點 Payoff 乘上發生機率，再從 T=2 一次折現回 T=0")
    
    # 組合公式: C(n, k)
    def nCr(n, r):
        return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

    weighted_sum = 0
    print("  計算細節：")
    
    # 我們重新遍歷一次 T=2 的情況
    for i in range(N + 1):
        # i 是上漲次數
        # 1. 該路徑發生的機率 (二項式機率): C(N, i) * p^i * q^(N-i)
        prob = nCr(N, i) * (p ** i) * (q ** (N - i))
        
        # 2. 該路徑的 Payoff (剛剛算過了)
        S_final = S0 * (u ** i) * (d ** (N - i))
        payoff = max(K - S_final, 0)
        
        # 加總
        weighted_sum += prob * payoff
        
        path_name = "UU" if i==2 else ("UD" if i==1 else "DD")
        print(f"    路徑 {path_name}: 機率 {prob:.4f} × Payoff {payoff:.1f} = {prob * payoff:.4f}")

    # 一次折現 2 年 (e^(-r * 2T))
    total_discount_factor = math.exp(-r * (N * T_step))
    final_value_formula = weighted_sum * total_discount_factor
    
    print(f"\n  期望值總和: {weighted_sum:.4f}")
    print(f"  總折現因子 (e^-rT): {total_discount_factor:.4f}")
    print(f"  計算: {weighted_sum:.4f} × {total_discount_factor:.4f}")
    print(f"  -> ★ 公式法最終價值: {final_value_formula:.4f}")

    # ==========================================
    # 結論比較
    # ==========================================
    print("="*60)
    print(f"【結論比較】")
    print(f"  方法一 (倒推法): {final_value_tree:.4f}")
    print(f"  方法二 (公式法): {final_value_formula:.4f}")
    print(f"  兩者誤差: {abs(final_value_tree - final_value_formula):.10f}")
    print("="*60)

# 執行主程式
if __name__ == "__main__":
    detailed_binomial_pricing()
```

----
## 選擇權定價進階：CRR 模型與參數設定

接下來要處理波動率的問題
在實務和學術界，我們不會隨便亂猜數字，而是使用一套標準的方法來決定 $u$ 和 $d$。這套方法來自著名的 CRR 模型 (Cox-Ross-Rubinstein Model)

透過台積電選擇權的實例，解答關於「波動率」與「買賣價差」的進階問題

----
### 第一部分：為什麼不能隨便決定 $u$ 和 $d$？
如果我們隨意設定，例如 $u=1.1, d=1.0$ (不跌)，會發生什麼事？
#### 1. 樹狀圖會「分岔」到無法收拾
在標準的二項式樹中，我們希望它是一個 「可重合樹」(Recombining Tree)。
也就是說：「先漲後跌」的價格，要等於「先跌後漲」的價格 ($S_{ud} = S_{du}$)
* 如果 $S \times u \times d = S$，那麼中間的節點就會重合。
* 如果 $S \times u \times d \neq S$，例如 $u=1.1, d=1.0$：
	* 先漲後跌：$20 \times 1.1 \times 1 = 22$
	* 先跌後漲：$20 \times 1 \times 1.1 = 22$
(註：雖然看似重合，但在更多步數時，若沒有嚴格的對稱關係，計算上會變得非常複雜且缺乏效率。)

#### 2. CRR 的解決方案
CRR 模型提出了一個最簡單且有效率的限制條件：$$d = \frac{1}{u}$$
這保證了不管樹走了幾步，節點都會漂亮地重合，不會無限擴散。
* 1-Step：有 2 個結局 ($S_u, S_d$)
* 2-Step：有 3 個結局 ($S_{uu}, S_{ud}, S_{dd}$)
* N-Step：會有 $N+1$ 個結局

### $u$ 和 $d$ 的黃金公式
既然知道了 $d = 1/u$，那 $u$ 到底該等於多少？它必須反映資產的「波動」程度。CRR 推導出了這個公式：
#### 1. 上漲因子公式
$$u = e^{\sigma \sqrt{\Delta t}}$$$$d = e^{-\sigma \sqrt{\Delta t}} = \frac{1}{u}$$這裡有兩個超級重要的參數，請務必搞懂：
#### A. 時間間隔 $\Delta t$ (Delta t)
這是樹狀圖「走一步」的時間長度，必須是 年化 (Annualized) 的單位。
公式：$\Delta t = \frac{\text{到期總時間 } T}{\text{步數 } N}$
* 注意：這不是指距離到期還有幾天，而是指你在模型中設定「多久跳動一次股價」。
* 如果總時間是 3 個月 ($0.25$ 年)，你的樹只走 1 步，那 $\Delta t = 0.25$。
* 如果總時間是 3 個月，你的樹要走 3 步，那 $\Delta t = 0.25 / 3$。

#### B. 波動率 $\sigma$ (Sigma / Volatility)
這是最容易犯錯的地方！
* 錯誤觀念：直接算「股價」的標準差。
* 正確觀念：必須計算 「報酬率」(Return) 的標準差。
	* 也就是 $\ln(S_t / S_{t-1})$ 的標準差
* 單位：同樣必須是 年化波動率
只要有了 $\sigma$ 和 $\Delta t$，代入公式，你就能得到科學且標準的 $u$ 和 $d$。

---- 

## 台積電選擇權定價
#### 1. 題目參數 (Input Data)
我們觀察到一檔台積電 (2330) 的選擇權（可能是權證或個股期權）：
![image](https://hackmd.io/_uploads/B1viZJzPZl.png)
* 標的股價 ($S_0$)：1035 元
* 履約價 ($K$)：943.48 元
* 剩餘到期日：33 天 $\rightarrow$ 年化大 $T = 33/365 \approx 0.0904$ 年
* 無風險利率 ($r$)：1.9% ($0.019$) —— 這是 Will 從多家券商報價反推出來的平均水準。
* 行使比例 (Ratio)：0.02 (這是權證或特定合約常見的條款，代表 1 張憑證對應 0.02 股)。

#### 2. 關於波動率 ($\sigma$) 的玄機
* 買價波動率 (Bid Vol)：53.85% ($0.5385$)
* 賣價波動率 (Ask Vol)：75.61% ($0.7561$)

為什麼有兩個？這牽涉到實務操作（稍後在 Q&A 詳細解釋）。我們先用這兩個數字分別計算。

#### 3. 開始計算 (Python 程式邏輯)
為了讓答案精準逼近 Black-Scholes 模型（也就是券商的電腦報價），我們需要把 步數 ($N$) 設得很大（例如 100 步或 1000 步）。
#### 步驟 A：算出 $u, d, p$假設我們用 買價波動率 53.85%，且設 $N=1000$ (假設值，為了逼近)。
* $\Delta t = 0.0904 / 1000$
* $u = e^{0.5385 \times \sqrt{\Delta t}}$
* $d = 1/u$
* $p = \frac{e^{r\Delta t} - d}{u - d}$

#### 步驟 B：算出理論價格 透過二項式模型的倒推法（或 Python 程式），我們算出了一單位的選擇權價格。
* 計算出的原始模型價格約為：120.8579 元
#### 步驟 C：調整行使比例 因為這個合約有行使比例限制 (0.02)。
最終價格 = $120.8579 \times 0.02 \approx \mathbf{2.417}$


#### 例子所用的買/賣價波動率是如何計算的？
* 歷史資料法 (Historical Volatility)：抓過去一段時間（如 30 天、90 天）的股價，算出每日報酬率，取標準差，再乘以 $\sqrt{252}$ (年化)。這代表「過去」的波動
* GARCH 模型：一種更高級的時間序列統計模型，用來預測波動率
* 隱含波動率 (Implied Volatility, IV)： 我們不是「算出」波動率，而是看市場上現在選擇權賣多少錢 (Market Price)，然後代入公式「反推」出市場共識的波動率是多少。影片中的例子，很可能就是網站已經幫你反推好的 IV

#### 為什麼會有「買價波動率」和「賣價波動率」兩個數值？$\sigma$ 不應該只有一個嗎？
理論上，同一檔股票在同一個瞬間，未來的波動應該只有一個真理（$\sigma$）。但在交易市場上：
* 選擇權會有 買價 (Bid Price) 和 賣價 (Ask Price)。通常 Ask > Bid (因為券商要賺價差)
* 在 Black-Scholes 或 Binomial 模型中，$S, K, T, r$ 都是固定的客觀事實。唯一能調整的變數就是 $\sigma$
* ：為了讓模型算出來的價格等於市場的 Bid Price，我們必須代入一個較低的 $\sigma$；為了等於 Ask Price，必須代入一個較高的 $\sigma$
	* 所以那兩個波動率，其實是 「隱含」在買價和賣價中的波動率。它們反映了市場對於流動性和風險的報價，而不僅僅是股票本身的波動

```
import math
from scipy.stats import norm  # 用於計算 Black-Scholes 的常態分佈累積機率

class BinomialTreeOption:
    def __init__(self, S0, K, T, r, sigma, N, option_type='call', exercise_style='european'):
        """
        初始化選擇權評價模型
        參數:
            S0 (float): 標的物當前價格
            K (float): 履約價 (Strike Price)
            T (float): 到期時間 (以年為單位)
            r (float): 無風險利率 (年化，小數點格式，例如 0.05)
            sigma (float): 波動率 (年化，小數點格式，例如 0.3)
            N (int): 二項式樹的步數 (Steps)
            option_type (str): 'call' (買權) 或 'put' (賣權)
            exercise_style (str): 'european' (歐式) 或 'american' (美式)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.N = N
        self.option_type = option_type.lower()
        self.exercise_style = exercise_style.lower()
        
        # --- 1. 計算 CRR 模型的關鍵參數 ---
        self.dt = T / N  # 每一步的時間長度 (年)
        self.df = math.exp(-r * self.dt)  # 單步折現因子
        
        # CRR 公式: u = exp(sigma * sqrt(dt)), d = 1/u
        self.u = math.exp(sigma * math.sqrt(self.dt))
        self.d = 1 / self.u
        
        # 風險中性機率 p
        self.p = (math.exp(r * self.dt) - self.d) / (self.u - self.d)
        
        # 檢查無套利條件 (d < e^rdt < u)
        if not (self.d < math.exp(r * self.dt) < self.u):
            print("警告: 步數過少或波動率極端，可能導致機率 p 超出 [0,1] 範圍。")

    def payoff(self, S):
        """計算到期時的內含價值 (Intrinsic Value)"""
        if self.option_type == 'call':
            return max(S - self.K, 0)
        else:
            return max(self.K - S, 0)

    def calculate_price(self):
        """
        執行二項式樹運算 (倒推法)
        """
        # --- 步驟 A: 初始化到期日 (Time = N) 的資產價格與選擇權價值 ---
        # asset_prices[i] 代表在到期時，上漲 i 次的股價
        # values[i] 代表對應的選擇權價值 (Payoff)
        values = []
        for i in range(self.N + 1):
            # 股價公式: S0 * u^i * d^(N-i)
            S_final = self.S0 * (self.u ** i) * (self.d ** (self.N - i))
            values.append(self.payoff(S_final))

        # --- 步驟 B: 逐步倒推回 Time = 0 ---
        # 從第 N-1 步推回第 0 步
        for step in range(self.N - 1, -1, -1):
            for i in range(step + 1):
                # 1. 計算「繼續持有」的期望值 (Continuation Value)
                # 使用風險中性機率 p 加權平均，再折現
                f_up = values[i + 1]   # 上漲節點的價值
                f_down = values[i]     # 下跌節點的價值
                hold_value = self.df * (self.p * f_up + (1 - self.p) * f_down)
                
                # 2. 判斷是否提早執行 (美式選擇權專用)
                if self.exercise_style == 'american':
                    # 計算該節點當下的股價
                    S_current = self.S0 * (self.u ** i) * (self.d ** (step - i))
                    exercise_value = self.payoff(S_current)
                    # 取兩者較大值
                    values[i] = max(hold_value, exercise_value)
                else:
                    # 歐式選擇權只能繼續持有
                    values[i] = hold_value

        # 最終 values[0] 即為現在的理論價格
        return values[0]

    def black_scholes_price(self):
        """
        計算 Black-Scholes 公式解 (僅供歐式選擇權對照)
        """
        d1 = (math.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)
        
        if self.option_type == 'call':
            price = self.S0 * norm.cdf(d1) - self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * math.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
        return price

# ==========================================================
#  主程式執行區：驗證台積電選擇權實例
# ==========================================================

print("="*60)
print("   二項式模型 (Binomial Model) - 台積電權證實例驗證")
print("="*60)

# --- 1. 設定參數 (來源：教學影片/文本) ---
# 標的: 台積電 (2330)
S_input = 1035          # 股價
K_input = 943.48        # 履約價
# 時間: 剩餘 33 天
T_input = 33 / 365.0    
# 利率: 1.9%
r_input = 0.019         
# 波動率: 買價波動率 53.85%
sigma_bid = 0.5385      
# 行使比例: 權證常見規格
ratio = 0.02            

# 設定步數: 為了逼近精準報價，我們設 N = 1000
steps = 1000

print(f"【輸入參數】")
print(f"  股價 (S): {S_input}")
print(f"  履約價 (K): {K_input}")
print(f"  年化時間 (T): {T_input:.4f} 年 ({33}天)")
print(f"  無風險利率 (r): {r_input*100}%")
print(f"  波動率 (Sigma): {sigma_bid*100}%")
print(f"  計算步數 (N): {steps}")
print(f"  行使比例: {ratio}")
print("-" * 60)

# --- 2. 建立模型物件 (買價 Bid Case) ---
model_bid = BinomialTreeOption(
    S0=S_input, 
    K=K_input, 
    T=T_input, 
    r=r_input, 
    sigma=sigma_bid, 
    N=steps, 
    option_type='call', 
    exercise_style='european' # 假設為歐式 (一般權證多為歐式)
)

# --- 3. 執行計算 ---
# 計算單一單位的選擇權價格
unit_price = model_bid.calculate_price()
# 乘上行使比例 (權證價格)
warrant_price = unit_price * ratio

# Black-Scholes 對照組
bs_unit_price = model_bid.black_scholes_price()
bs_warrant_price = bs_unit_price * ratio

print(f"【計算結果 - 買價 (Bid)】")
print(f"  1. 模型原始價格 (Unit Price): {unit_price:.4f}")
print(f"  2. 權證理論價格 (x {ratio}):   {warrant_price:.4f}")
print(f"     (影片中市場報價為 2.4，模型計算結果為 2.4172，非常接近)")
print(f"  3. Black-Scholes 對照:        {bs_warrant_price:.4f}")
print(f"     (二項式模型與 BS 模型的誤差: {abs(warrant_price - bs_warrant_price):.6f})")

print("-" * 60)

# --- 4. 額外測試：賣價 (Ask Case) ---
# 影片中提到賣價波動率為 75.61%，市場報價為 2.85
sigma_ask = 0.7561

model_ask = BinomialTreeOption(
    S0=S_input, K=K_input, T=T_input, r=r_input, 
    sigma=sigma_ask, N=steps, option_type='call'
)

ask_unit_price = model_ask.calculate_price()
ask_warrant_price = ask_unit_price * ratio

print(f"【計算結果 - 賣價 (Ask)】")
print(f"  輸入波動率: {sigma_ask*100}%")
print(f"  權證理論價格: {ask_warrant_price:.4f}")
print(f"  (影片中市場報價為 2.85，模型計算結果為 2.8722，驗證成功)")
print("="*60)
```