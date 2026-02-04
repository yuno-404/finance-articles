---
title: 凱利公式 (Kelly Criterion)

---

## 凱利公式 (Kelly Criterion)

今天要特別來寫一篇關於凱利公式的文章
內容主要來自於 《Finance with Monte Carlo》 一書
這一章節要是在回答 「即便我知道這場遊戲對我有利（正期望值），我到底該下注多少比例？
目的不是為了最大化「下一把」賺多少錢（那是算術平均期望值），而是為了最大化「長期資產的複利增長速度」（幾何平均增長率），如果你每次都全押（$f=1$），只要輸一次就破產了。如果你押太少（$f \approx 0$），資產增長太慢。凱利公式就是在尋找中間那個「增長最快」的甜蜜點 (Sweet Spot)。


----
## 簡單遊戲 (The Simple Game)
凱利公式最基礎的形式，適用於只有兩種結果（贏或輸）的賭局或投資，直覺想法就是優勢越大，下注越大；風險越高，下注越小。
中間都是計算過程，懶得看可以往下滑

1. 問題設定
	* 勝率：贏的機率為 $p$，輸的機率為 $q = 1-p$ 。
	* 賠率 (Gain)：若贏，淨獲利為賭注的 $\gamma$ 倍（例如獲利 88%，則 $\gamma = 0.88$）；若輸，損失全部賭注 。
	* 期望值：單位賭注的期望收益為 $E_{gain} = \gamma p - q$。若期望值為正，才值得下注 。
	* 決策變數：我們每次下注資金的一個固定比例 $f$ 

2. 財富演變推導
假設初始財富為 $F_0$
	* 贏的情況：財富變為 $F_0 + \gamma f F_0 = (1 + \gamma f)F_0$
	* 	輸的情況：財富變為 $F_0 - f F_0 = (1 - f)F_0$
	
經過 $N$ 次下注後，假設贏了 $W$ 次，輸了 $L$ 次，最終財富 $F_N$ 為：
$$F_N = (1 + \gamma f)^W (1 - f)^L F_0$$

3. 最大化增長率 (Maximizing Growth Rate)
凱利的核心觀點是最大化「指數增長率」$G_N$。
根據公式 $e^{G_N N} F_0 = F_N$，我們取對數：

$$G_N = \frac{1}{N} \log \left( \frac{F_N}{F_0} \right) = \frac{W}{N} \log(1 + \gamma f) + \frac{L}{N} \log(1 - f)$$

當 $N$ 趨近無窮大時，根據大數法則，$\frac{W}{N} \to p$ 且 $\frac{L}{N} \to q$。因此，期望增長率 $G$ 為：
$$G(f) = p \log(1 + \gamma f) + q \log(1 - f)$$

4. 求解最佳 $f$ (Optimization)
為了找出使 $G$ 最大的 $f$，我們對 $f$ 微分並令其為 0：$$\frac{dG}{df} = \frac{p \gamma}{1 + \gamma f} - \frac{q}{1 - f} = 0$$
通分整理：
$$p \gamma (1 - f) - q (1 + \gamma f) = 0$$$$p \gamma - p \gamma f - q - q \gamma f = 0$$$$p \gamma - q = f \gamma (p + q)$$
因為 $p+q=1$，所以：$$f^* = \frac{p \gamma - q}{\gamma}$$這就是著名的凱利公式。它告訴我們，最佳下注比例等於「期望淨收益除以贏的賠率」 。
$$f = \frac{E_{gain}}{\gamma}$$
* $f$：你應該下注的資金比例（最佳下注比例）。
* $E_{gain}$ (期望淨收益)：分子。代表這場賭局平均而言對你有多「有利」。
	* 如果 $E_{gain}$ 很高（例如你掌握了內線消息，勝率極高），分子變大，公式告訴你可以下重注
	* 如果 $E_{gain}$ 是 0 或負數（例如賭場裡的輪盤），分子為 0 或負，公式告訴你 $f \le 0$，也就是完全不該下注 
* $\gamma$ (贏的賠率)：分母。代表你贏的時候，每 1 元賭注能淨賺多少錢（賠付倍數）。
	* 為什麼要「除以」賠率？這有點反直覺。通常賠率高（例如 1 賠 100）不是很好嗎？
	* 但通常高賠率伴隨著低勝率（High Payout, Low Probability）。如果 $\gamma$ 很大，分母變大，會縮小你的下注比例 $f$


因為高賠率的遊戲通常意味著你會經歷很多次的「輸」，為了避免在贏那把大錢之前就先破產，你必須縮小每次的下注比例

範例:
* 贏的賠率 ($\gamma$)：0.88 (贏了賺 88%)。
* 期望淨收益 ($E_{gain}$)：計算出來是 0.128 (平均每下 1 元賺 0.128 元)
* 計算：$0.88 \times 60\% (\text{贏}) - 1 \times 40\% (\text{輸}) = 0.128$
$$f = \frac{0.128 \text{ (期望淨收益)}}{0.88 \text{ (贏的賠率)}} = 0.145$$


原則就是，下注規模應該「與你的優勢成正比，與賭局的賠率成反比」。
這是簡單遊戲告訴我們的，接下來會越來越複雜。

----
## 含巨額損失的簡單遊戲 (Catastrophic Loss)
現實金融市場中（如賣出選擇權），損失可能超過本金。本節修正公式以適應此情況。
假設今天交易非常多次的情況下，有幾次虧損特別大，如果使用平均方式去計算虧損，那是不是會低估風險?
這裡的說法是，雖然平均虧損會因此虧損特別大的交易而提高，但不會真的跟毀滅性的虧損一樣大，那當今天真的遇到崩跌，會直接死掉，因為你下注太大了!
因此要將這些毀滅性的崩跌虧損挑出來計算，不能跟其他一般性的虧損平均。
* 假設最大虧損是普通單位的 1.93 倍（$\mu = 1.93$）
* 下注比例 $f$ 絕對不能超過 $\frac{1}{1.93} \approx 51.8\%$

了解這裡要講的概念後，開始往下仔細說明內容

* 賺錢 (Wins)：49 次，平均每筆賺 $889.12
* 虧錢 (Losses)：共 26 次，但作者將其細分為兩種：
	* 普通虧損 (Intermediate Loss)：24 次，平均每筆賠 $1,366.32。
	* 毀滅性虧損 (Catastrophic Loss)：最慘的 2 次，平均每筆賠 $2,637.00

凱利公式通常不直接使用「金額（美元）」計算，而是使用「單位（Units）」。這樣可以把公式標準化。
我們把「普通虧損」的金額（$1,366.32）當作「1 個單位的賭注」。這意味著如果發生普通虧損，你的損失就是 $1 \times f$（$f$ 是下注比例）。

1. 問題設定：
	* 贏的賠率 ($\gamma$)：賺的錢是普通虧損的幾倍？$$\gamma = \frac{889.12}{1366.32} \approx 0.651$$
意義：每承擔 1 單位的普通風險，獲利時可以賺 0.651 單位
	* 毀滅性虧損倍數 ($\mu$)：
大賠的錢是普通虧損的幾倍？$$\mu = \frac{2637.00}{1366.32} \approx 1.930$$
	意義：遇到大賠時，你會損失 1.93 個單位的賭注
	
計算各個機率 ($p$) 
* 贏的機率 $p_1 = 49 / 75 \approx 0.653$
* 普通輸機率 $p_2 = 24 / 75 = 0.32$
* 大賠機率 $p_3 = 2 / 75 \approx 0.026$
	
### 檢查期望值 (Expected Gain)
在運用凱利公式前，必須確定這場遊戲是「正期望值」的（值得玩）。如果期望值是負的，最佳策略就是不玩 ($f=0$)。
計算單位期望值 ：
$$E_{gain} = (\text{贏的獲利} \times p_1) - (\text{普通虧損} \times p_2) - (\text{大賠虧損} \times p_3)$$$$E_{gain} = (0.651 \times 0.653) - (1 \times 0.32) - (1.930 \times 0.026)$$
(註：普通虧損倍數為 1)
$$E_{gain} = 0.425 - 0.32 - 0.050 = 0.055$$

### 設定安全限制 (Constraint)
這是與普通凱利公式最不同的一點。
在普通賭局，輸了就是賠光籌碼（損失 1 倍），所以你可以下注 100% ($f=1$) 頂多歸零。但在這裡，如果遇到「毀滅性虧損」，你會賠掉 1.93 倍 的籌碼。
如果你下注 $f=1$ (100% 資金)，一旦遇到大賠，資產變為：$$1 - (1.93 \times 1) = -0.93$$
你會負債，直接破產出場。因此必須有嚴格限制 ：
為了保證活著（資產 $> 0$），下注比例 $f$ 必須滿足：
$$1 - \mu f > 0$$$$f < \frac{1}{\mu}$$$$f < \frac{1}{1.93} \approx 0.518$$你的下注比例絕對不能超過資金的 51.8%。


### 推導過程增長率函數 $G(f)$：
我們要最大化長期財富增長率 $G$。根據複利原理，經過 $N$ 次交易後的財富公式為 ：
$$F_N = (1 + \gamma f)^W (1 - f)^L (1 - \mu f)^C F_0$$$W$: 贏的次數$L$: 普通輸的次數$C$: 大賠的次數取對數算平均增長率 $G(f)$ ：
$$G(f) = p_1 \ln(1 + \gamma f) + p_2 \ln(1 - f) + p_3 \ln(1 - \mu f)$$這條公式的意思是：加權平均你的對數資產變化。
### 微分解方程式 (最困難的步驟)
為了找到讓 $G(f)$ 最大的 $f$，我們需要對 $f$ 微分，並令導數為 0。
1. 微分 $G'(f)$ ：
$$G'(f) = \frac{p_1 \gamma}{1 + \gamma f} - \frac{p_2}{1 - f} - \frac{p_3 \mu}{1 - \mu f} = 0$$
中間是減號，因為 $\ln(1-f)$ 的微分有連鎖律出來的負號
2. 展開求解 ：
我們要解這個方程式。為了消去分母，我們將等式同乘 $(1+\gamma f)(1-f)(1-\mu f)$：
$$p_1 \gamma (1-f)(1-\mu f) - p_2 (1+\gamma f)(1-\mu f) - p_3 \mu (1+\gamma f)(1-f) = 0$$
這是一個關於 $f$ 的一元二次方程式 ($Af^2 + Bf + C = 0$)。讓我們把它展開來看看項次結構：
* 第一項 ($p_1$ 部分): $p_1 \gamma (1 - \mu f - f + \mu f^2)$
* 第二項 ($p_2$ 部分): $p_2 (1 - \mu f + \gamma f - \gamma \mu f^2)$
* 第三項 ($p_3$ 部分): $p_3 \mu (1 - f + \gamma f - \gamma f^2)$
把這些係數整理起來（這通常由電腦做，手算極其繁瑣），你會得到一個二次方程式。
3. 代入數值求解： 讓我們使用書中的數值：
代入數值求解：讓我們使用書中的數值：
* $p_1=0.653, \gamma=0.651$
* $p_2=0.32$
* $p_3=0.026, \mu=1.93$
將數值代入 $G'(f)=0$ 求解。
$$f \approx 0.078$$


你應該拿出總資金的 7.8% 去做每一筆蝴蝶價差交易。
如果不考慮毀滅性虧損（把 $\mu$ 當作 1），算出來的 $f$ 會大很多
但因為我們考慮了那個發生機率雖低（2.6%）但後果嚴重（賠 1.93 倍）的黑天鵝事件，數學模型自動把最佳下注比例從潛在的高水位「拉低」到了 7.8%
7.8% 遠小於我們在第四步算出的破產上限 51.8% ($0.078 < 0.518$)

----
## 單一贏家的最佳配置 (One Winner Allocation)
當我有好幾個選擇（A、B、C...），但只有其中一個會贏，我該怎麼分配資金，又要保留多少現金？

* 從單一到多重：之前的章節是講把資金分配給「單一」冒險 。現在我們要考慮「多個」冒險 。
* 互斥條件 (Mutually Exclusive)：假設這些冒險中，只有一個會成功，其他的投資都會歸零 。
	* 賽馬比喻：這個問題最原始的凱利公式形式，其實是用來解決賽馬下注問題的（只有一匹馬會贏）。

### 從 Gain ($\gamma$) 到 Payout ($\alpha$)
為了方便處理多重選擇的數學運算，這裡做了一個變數轉換。
變數定義：
* Gain ($\gamma$)：淨獲利（扣掉本金）。
* Payout ($\alpha$)：賠付倍數（包含本金）。
* 
期望值 = (贏的機率 $\times$ 淨賺) - (輸的機率 $\times$ 賠掉本金)
公式：$E = p \times \gamma - (1-p) \times 1$
平均拿回來的錢 = 贏的機率 $p$ $\times$ 拿回總額 $\alpha$
公式：$E = p\alpha - 1$ 
關係式：$$\alpha = \gamma + 1$$
理由：因為 Payout 包含了你付出的 1 元本金 。
判斷有利的條件：
* 賭局有利的定義是期望值大於 0：$E = p\gamma - (1-p) > 0$ 。
* 換成 Payout 表示，條件變為：$p\alpha > 1$ 。
* 證明：$p\alpha - 1 = p(\gamma + 1) - 1 = p\gamma + p - 1 = p\gamma - (1-p)$。
* 如果 $p\alpha = 1$，代表這是公平賭局 (Fair odds) 。

傳統說法：期望值大於 0 才玩。$$E > 0$$
新說法：期望賠付大於 1 才玩。$$p\alpha > 1$$

#### 四選一的問題 (A Four Choice Allocation Problem)
選項：有 A, B, C, D 四個投資標的。
* $f_i$：分配給投資 $i$ 的資金比例 ($i = A, B, C, D$)。
* $b$：保留現金的比例 (Fraction held back)，也就是不投入賭局的部分 。

限制條件：所有的錢加起來必須等於 1。$$b + f_A + f_B + f_C + f_D = 1$$
* A: 勝率 0.5, 賠付 2.1
* B: 勝率 0.3, 賠付 3.2
* C: 勝率 0.1, 賠付 10.8
* D: 勝率 0.1, 賠付 8.5
#### $\tau$ (Tau) 的定義與意義
在解題之前，作者引入了一個非常重要的指標 $\tau$。
定義：所有賠付倍數的倒數和。$$\tau = \sum \frac{1}{\alpha_i}$$

這代表了賭局的「公平性」或「莊家優勢」
* $\tau = 1$：代表所有收進來的賭注，莊家都賠出去了 (All money paid in is paid back out) 。這意味著沒有手續費或莊家抽成。
* $\tau > 1$：賠出去的錢少於收進來的錢 。這代表有交易成本、手續費或莊家優勢。這是現實金融市場最常見的情況。
* $\tau < 1$：這場冒險整體是在「創造金錢」。

#### 推導增長率公式 (The Growth Rate Equation)
Step 1: 單次財富變化 
假設初始財富是 $F_0$
如果 A 贏了：
* 你手上的現金 $b F_0$ 還在
* 你在 A 的投資變成了 $\alpha_A f_A F_0$
* 你在 B, C, D 的投資歸零了。
* 所以新的財富是：$(b + \alpha_A f_A) F_0$。

同理，如果 B 贏了，財富變為 $(b + \alpha_B f_B) F_0$，以此類推

Step 2: N 次後的財富 
經過 $N$ 次投資後，假設 A 贏了 $N_A$ 次，B 贏了 $N_B$ 次...
$$F_N = (b + \alpha_A f_A)^{N_A} (b + \alpha_B f_B)^{N_B} (b + \alpha_C f_C)^{N_C} (b + \alpha_D f_D)^{N_D} F_0$$
Step 3: 取對數求增長率 G 
我們要求的是指數增長率 $G$。利用公式 $G = \lim_{N \to \infty} \frac{1}{N} \log \frac{F_N}{F_0}$。
因為當 $N$ 趨近無限大時，$\frac{N_A}{N} \to p_A$ (勝率)，所以公式變為：
$$G = p_A \log(b + \alpha_A f_A) + p_B \log(b + \alpha_B f_B) + p_C \log(b + \alpha_C f_C) + p_D \log(b + \alpha_D f_D)$$

這就是我們要最大化的目標函數

## 使用拉格朗日乘數法求解 (Optimization)
這是一個「有條件限制的極值問題」，因為我們必須滿足 $b + \sum f_i = 1$。所以作者使用了拉格朗日乘數法 (LaGrange Method) 
我們對每個變數 ($f_i$ 和 $b$) 偏微分，並令其等於拉格朗日乘數 $\lambda$
對每個投資 $f_i$ 微分：
$$\frac{\partial G}{\partial f_i} = \frac{p_i \alpha_i}{b + \alpha_i f_i} = \lambda \quad (i = A, B, C, D)$$(直觀意義：這是該投資的「邊際效益」。最佳化時，每一分錢投入任何地方的邊際效益應該相等。)
對保留現金 $b$ 微分：$$\frac{\partial G}{\partial b} = \sum_{i} \frac{p_i}{b + \alpha_i f_i} = \lambda$$
限制條件：$$b + f_A + f_B + f_C + f_D = 1$$

### 什麼是「互斥」？
假設有四支股票（或四匹馬）A、B、C、D 競爭同一個合約（或冠軍）。
* 規則：最後只有 一個 會贏。A 贏了，B、C、D 就一定輸。
* 資金限制：你手上的錢總共只有 $1$ (100%)。
* 分配：你必須決定要把多少錢下注在 A ($f_A$)、B ($f_B$) ... 以及 保留多少現金 ($b$)。
	* 公式：$b + f_A + f_B + f_C + f_D = 1$ 

#### 賠付 ($\alpha$) 與 公平性 ($\tau$)
為了計算方便，我們不看淨利，改看 賠付倍數 (Payout, $\alpha$)。
$\alpha = \text{賠率} + 1$ (包含拿回的本金)。
指標 $\tau$ (Tau)：莊家有沒有抽水？ 
$$\tau = \sum \frac{1}{\alpha_i} = \frac{1}{\alpha_A} + \frac{1}{\alpha_B} + \dots$$
* $\tau = 1$：公平賭局。莊家收多少賠多少。這時我們不需要保留現金 ($b=0$)，全力下注。
* $\tau > 1$：這場賭局莊家有優勢（手續費高）。這時我們 必須保留現金 ($b > 0$)，不能梭哈。

#### 目標函數
我們要最大化長期增長率 $G$ ：
$$G = p_A \ln(b + \alpha_A f_A) + p_B \ln(b + \alpha_B f_B) + \dots$$
(意思是：A 贏的機率 $\times$ A 贏時的資產對數 + B 贏的機率 $\times$ B 贏時的資產對數...)

#### 拉格朗日法 (Lagrange Multipliers) 的結論
這是一個「有限制條件」的極值問題。作者透過微分推導出一個 「黃金方程式」：
對於每一個被選中的投資 $i$（只要 $f_i > 0$），都必須滿足：$$b + \alpha_i f_i = p_i \alpha_i$$

這個公式非常重要！我們可以把它移項，得到 單一投資比例的計算公式：$$\alpha_i f_i = p_i \alpha_i - b$$$$f_i = p_i - \frac{b}{\alpha_i}      (公式 A)$$

這告訴我們：**只要算出了保留現金 $b$**，所有的下注比例 $f$ 就全部解開了！
####  推導 $b$ 的公式
既然我們知道總資金是 1，我們可以把上面的 公式 全部加起來：$$1 = b + f_A + f_B + f_C + \dots$$
將 (公式 A) 代入 $f$：
$$1 = b + (p_A - \frac{b}{\alpha_A}) + (p_B - \frac{b}{\alpha_B}) + (p_C - \frac{b}{\alpha_C}) + \dots$$
我們把有 $b$ 的項全部整理到一邊，沒 $b$ 的項整理到另一邊：
1. 拆開括號：
$$1 = b + p_A + p_B + p_C - b(\frac{1}{\alpha_A} + \frac{1}{\alpha_B} + \frac{1}{\alpha_C})$$
1. 移項 (把 $p$ 移到左邊)：
$$1 - (p_A + p_B + p_C) = b - b(\frac{1}{\alpha_A} + \frac{1}{\alpha_B} + \frac{1}{\alpha_C})$$
1. 提出 $b$：
$$1 - \sum p = b \left[ 1 - \sum \frac{1}{\alpha} \right]$$

1. 解出 $b$：
$$b = \frac{1 - \sum p}{1 - \sum \frac{1}{\alpha}}(公式 B)$$

#### 範例
* A: 勝率 $p=0.5$, 賠付 $\alpha=2.1$
* B: 勝率 $p=0.3$, 賠付 $\alpha=3.2$
* C: 勝率 $p=0.1$, 賠付 $\alpha=10.8$
* D: 勝率 $p=0.1$, 賠付 $\alpha=8.5$
#### 步驟 1：誰是好投資？ (Ranking)
計算期望值 ($p \times \alpha$) 看看誰大於 1：
* A: $0.5 \times 2.1 = 1.05$ (好)
* C: $0.1 \times 10.8 = 1.08$ (最好)
* B: $0.3 \times 3.2 = 0.96$ (差，期望值 < 1)
* D: $0.1 \times 8.5 = 0.85$ (很差)
#### 步驟 2：篩選組合 (Iterative Calculation)
這裡有一個反直覺的地方：即使 B 的期望值小於 1，它也可能被選入組合中！ 為什麼？因為它可以降低整體波動（避險作用）。我們必須用數學來驗證。
嘗試 1：只選最好的 A 和 C我們先假設只投 A 和 C。用 (公式 B) 算 $b$：
* 分子：$1 - (0.5 + 0.1) = 0.4$
* 分母：$1 - (\frac{1}{2.1} + \frac{1}{10.8}) = 1 - (0.476 + 0.092) = 1 - 0.568 = 0.432$
* $b = 0.4 / 0.432 \approx 0.925$

這時我們檢查被排除的 B。如果我們把 B 加進來，它的下注比例 $f_B$ 會大於 0 嗎？
用 (公式 A) 檢查：
$f_B = p_B - b / \alpha_B = 0.3 - 0.925 / 3.2 = 0.3 - 0.289 = \mathbf{0.011} > 0$
結果是正的！這代表加入 B 能讓組合更好，即使 B 單獨看是賠錢的
嘗試 2：加入 A, B, C (正確組合)既然 B 也值得投，我們重新計算包含 A、B、C 的 $b$ 。
* 分子 ($1 - \text{總勝率}$):$1 - (0.5 + 0.3 + 0.1) = 1 - 0.9 = \mathbf{0.1}$
* 分母 ($1 - \text{倒數賠率和}$):
$1 - (\frac{1}{2.1} + \frac{1}{3.2} + \frac{1}{10.8})$
$= 1 - (0.47619 + 0.3125 + 0.09259)$
$= 1 - 0.88128 = \mathbf{0.11872}$
* 算出最終保留現金 $b$：
$b = 0.1 / 0.11872 \approx \mathbf{0.8423}$ (84.23%)

#### 步驟 3：算出各別下注比例 $f$
現在我們有 $b = 0.8423$，直接代入 (公式 A) $f_i = p_i - b/\alpha_i$ ：
* 投資 A：$f_A = 0.5 - (0.8423 / 2.1) = 0.5 - 0.4011 = \mathbf{0.099}$ (9.9%)
* 投資 B (注意這個神奇的結果)：$f_B = 0.3 - (0.8423 / 3.2) = 0.3 - 0.2632 = \mathbf{0.037}$ (3.7%)這證明了，為了保護資產增長曲線，凱利公式建議你買一點「期望值略低於 1」的選項作為保險。
* 投資 C：$f_C = 0.1 - (0.8423 / 10.8) = 0.1 - 0.0780 = \mathbf{0.022}$ (2.2%)
* 投資 D (檢查用)： $f_D = 0.1 - (0.8423 / 8.5) = 0.1 - 0.099 = \mathbf{0.001}$ (趨近於 0) 書中直接設為 0，因為它的效益太低，且計算誤差可能導致微負值。

#### **凱利公式在多重選擇時**
* 會告訴你留多少錢 ($b=84\%$)
* 會自動分配資金：賠率好且穩的 A 拿最多 (9.9%)，高賠率但難中的 C 拿一點 (2.2%)
* 會做避險：即使 B 單獨看是不划算的，但為了在 A 和 C 都沒中時能補血，公式叫你也要買一點 B (3.7%)

----
## 多重贏家 (Multiple Winners)
在真實的股市中，台積電上漲（A贏），不代表聯發科一定會跌（B輸）。它們可能一起漲，也可能一起跌，或者一漲一跌。這就是所謂的「多重贏家 (Multiple Winners)」問題。

首先，我們必須搞清楚這場遊戲的所有可能結局。假設資產 A 和 B 是獨立的（A 的漲跌跟 B 無關），我們會有 $2 \times 2 = 4$ 種可能的結局。
 1. 參數設定 
*  資產 A：勝率 $p_A = 0.6$，賠付倍數 $\alpha_A = 2.1$
*  資產 B：勝率 $p_B = 0.5$，賠付倍數 $\alpha_B = 2.2$
*  資金分配：
	* $f_A$：投資 A 的比例
	*  $f_B$：投資 B 的比例
	*  $b$：保留現金的比例 ($b = 1 - f_A - f_B$)

我們要列出所有可能發生的情況，以及在該情況下你的財富會變成原本的幾倍：
#### 結局一：大獲全勝 (Both Succeed)
* 發生條件：A 贏 且 B 贏。
* 機率：$p_A \times p_B = 0.6 \times 0.5 = 0.3$。
* 財富倍數：保留的錢 ($b$) 還在，A 賺錢回來 ($\alpha_A f_A$)，B 也賺錢回來 ($\alpha_B f_B$)
* 總資產變為：$b + \alpha_A f_A + \alpha_B f_B$ 倍

#### 結局二：A 贏 B 輸 (A Succeeds, B Fails)
* 發生條件：A 贏 且 B 輸。
* 機率：$p_A \times (1 - p_B) = 0.6 \times 0.5 = 0.3$
* 財富倍數：保留的錢 ($b$) 還在，A 賺錢 ($\alpha_A f_A$)，但投在 B 的錢沒了
* 總資產變為：$b + \alpha_A f_A$ 倍
#### 結局三：A 輸 B 贏 (A Fails, B Succeeds)
* 發生條件：A 輸 且 B 贏。
* 機率：$(1 - p_A) \times p_B = 0.4 \times 0.5 = 0.2$
* 財富倍數：保留的錢 ($b$) 還在，B 賺錢 ($\alpha_B f_B$)，但投在 A 的錢沒了
* 總資產變為：$b + \alpha_B f_B$ 倍

#### 結局四：慘敗 (Both Fail)
* 發生條件：A 輸 且 B 輸。
* 機率：$(1 - p_A) \times財富倍數：只剩下保留的現金 $b$
* 財富倍數：只剩下保留的現金 $b$
* 總資產變為：$b$ 倍

### 建立增長率方程式 (The Growth Rate Equation)
凱利公式的目標是最大化長期增長率 ($G$)。這個 $G$ 其實就是上述四種結局的「對數期望值」。
$$G = \underbrace{p_A p_B \log(b + \alpha_A f_A + \alpha_B f_B)}_{\text{結局1: 雙贏}} + \underbrace{p_A (1-p_B) \log(b + \alpha_A f_A)}_{\text{結局2: A贏B輸}} + \underbrace{(1-p_A) p_B \log(b + \alpha_B f_B)}_{\text{結局3: A輸B贏}} + \underbrace{(1-p_A)(1-p_B) \log(b)}_{\text{結局4: 雙輸}}$$
我們要調整 $b, f_A, f_B$ 讓這個 $G$ 越大越好

### 求解方程式
使用「拉格朗日乘數法」將問題簡化
#### 變數 $w$ 的方程式
經過繁複的微分推導（書中省略了中間過程，直接給出簡化後的方程式），所有的變數都可以被濃縮成一個未知數 $w$
$$\frac{p_A p_B}{w + \tau_2 - 1} + \frac{p_A (1-p_B)}{w + \frac{1}{\alpha_B} - 1} + \frac{(1-p_A)p_B}{w + \frac{1}{\alpha_A} - 1} + \frac{(1-p_A)(1-p_B)}{w} = 0$$
* 這裡的 $\tau_2 = \frac{1}{\alpha_A} + \frac{1}{\alpha_B}$ 

#### 代入數值求解
使用數據代入，得到解：$w = 0.29113$ 
#### 還原出最佳投資比例
一旦算出了 $w$，就可以像解鎖一樣，把 $b, f_A, f_B$ 全部算出來：
* 第一步：算保留現金 $b$，我們可以推算出 $b$。
	* 計算結果：$b = 0.687$ (保留 68.7% 的現金)
* 第二步：算投資比例 $f_A, f_B$利用中間變數 $u, v$（由 $w$ 算出）
	* $u = 1 - \frac{1}{2.2} - 0.29113 = 0.2543$ 
	* $v = 1 - \frac{1}{2.1} - 0.29113 = 0.2327$ 
	* 最後得出：
		* $f_A = 0.2345$ (投 23.45% 在 A) 
		* $f_B = 0.0784$ (投 7.84% 在 B) 

#### 結論
* 驗算：$0.687 + 0.2345 + 0.0784 \approx 1.0$。符合資金總和為 1 的限制 
* 這個配置下的增長率 $G = 0.0347$
經過 60 次交易後，預期財富倍數 $E(F_{60}) = e^{60 \times G} \approx 8.02$ 倍 

當你同時買進兩支獨立的股票時，不能只是把它們當作分開的賭局來算。 你必須考慮它們互動的所有四種可能性（一起漲、一起跌、一漲一跌）
* 不要梭哈：即使兩個期望值都是正的
* 重倉強者：A 的勝率高 (0.6 vs 0.5)，所以分配 23.5% 給它
* 輕倉弱者：B 雖然期望值也是正的，但為了平衡風險，只分給它 7.8%


### Allocation for Correlated Investments
但現實沒那麼多獨立事件，如果兩支股票 A 和 B 會一起漲跌（有相關性），原本的凱利公式要怎麼修改？
我們必須重新計算四種機率。作者引入了 相關係數 (Correlation, $\rho$) 來修正機率
#### 設定隨機變數 $X, Y$
*  令 $X$ 代表 A 的結果：贏=1，輸=0。
*  令 $Y$ 代表 B 的結果：贏=1，輸=0
*  平均值 (期望值)：$E[X] = p_A$
*  標準差：$\sigma_A = \sqrt{p_A(1-p_A)}$ (這是伯努利分佈的標準差公式)

####  推導「雙贏機率」 $d(1,1)$
我們想知道 $d(1,1)$，也就是 $P(X=1, Y=1)$
利用統計學公式：相關係數定義
$$\rho = \frac{Cov(X,Y)}{\sigma_A \sigma_B}$$
移項得到 共變異數 (Covariance)：$$Cov(X,Y) = \rho \sigma_A \sigma_B$$

另一方面，共變異數的定義是：$$Cov(X,Y) = E[XY] - E[X]E[Y]$$
這裡有個技巧：$E[XY]$ 是什麼？
因為 $X, Y$ 只有 0 或 1。只有當 $X=1$ 且 $Y=1$ 時，$X \times Y$ 才會是 1，其他情況都是 0。
所以 $E[XY]$ 其實就是「雙贏的機率」 $d(1,1)$
將這些代回去：$$\rho \sigma_A \sigma_B = d(1,1) - p_A p_B$$移項求出 $d(1,1)$ (修正後的雙贏機率)：$$d(1,1) = p_A p_B + \rho \sigma_A \sigma_B$$

**雙贏的機率 = 「原本獨立的機率」+「因為相關性而增加的機率」。**
既然算出了雙贏，其他三種情況可以用「加減法」推出來（因為總和必須守恆）
A贏 B輸 $d(1,0)$：A 贏的總機率是 $p_A$。A 贏的情況只有兩種：「A贏B贏」或「A贏B輸」。所以：$p_A = d(1,1) + d(1,0)$$$d(1,0) = p_A - d(1,1)$$代入剛算的公式：$$d(1,0) = p_A - (p_A p_B + \rho \sigma_A \sigma_B) = p_A(1 - p_B) - \rho \sigma_A \sigma_B$$

同理可推導出 B贏 A輸 $d(0,1)$ 和 雙輸 $d(0,0)$

#### 解方程式 (The Equation)
現在我們有了修正後的四個機率：$d(1,1), d(1,0), d(0,1), d(0,0)$。
接下來的步驟跟上一節一模一樣，只是把公式裡的 $p_A p_B$ 換成 $d(1,1)$ 而已
目標是解出那個神奇數字 $w$
$$0 = \frac{d(1,1)}{w + \tau_2 - 1} + \frac{d(1,0)}{w + \frac{1}{\alpha_B} - 1} + \frac{d(0,1)}{w + \frac{1}{\alpha_A} - 1} + \frac{d(0,0)}{w}$$
算出 $w$ 後，再回推 $b, f_A, f_B$。
### 三種情境的結果分析
作者透過改變 $\rho$ 值，展示了相關性如何劇烈改變你的投資組合。這是這一章的精華。
#### 情境 1：低度正相關 ($\rho = 0.3$)
* 狀況：兩支股票有點連動。
* 結果：
	* 保留現金 $b$ 增加：從原本獨立時的 68.7% 增加到 75.5%。(因為風險變大了，不能押那麼多)
	* 弱者被犧牲：B 的下注比例 $f_B$ 從 7.8% 暴跌到 1.2%
	* 強者持平：A 的比例 $f_A$ 差不多 (23% $\to$ 22.9%)
* 如果資產有正相關，凱利公式會叫你砍掉弱的那支，因為分散風險的效果變差了

#### 情境 2：高度正相關 ($\rho = 0.8$) 
* 狀況：兩支股票幾乎同進同出。
	* 如果你試著去解上面的 $w$ 方程式，會發現算出來的 $f_B$ 變成負的。但在這模型中我們不能放空，所以邊界解發生了——直接令 $f_B = 0$
	* 既然 $f_B=0$，這問題就退化成「只買 A 的單一賭局」。
	* 公式變回最簡單的：$f_A = \frac{p_A \alpha_A - 1}{\alpha_A - 1}$。
	* 算出 $f_A = 23.6\%$
* 當相關性太高時，分散投資完全無效。 直接把錢全部押在期望值最高的那支 (A) 就好，B 完全不用買
#### 情境 3：負相關 ($\rho = -0.3$) 
* 狀況：A 漲的時候 B 容易跌（例如股票 vs 債券，或是航空股 vs 油價）
	* 雙贏機率 $d(1,1)$ 減少，但一贏一輸機率增加。這代表你的總資產波動變小了（互補）
	* 敢下重注：保留現金 $b$ 大幅下降到 56.9%（原本要留 75%）
	* 倉位大增：A ($27.9\%$) 和 B ($15.3\%$) 都比獨立時買得更多
	* 賺更多：增長率 $G$ 飆升到 0.0449
* 負相關是凱利公式最喜歡的屬性。 它提供了天然的避險，讓你敢開更大的槓桿，長期賺更多的錢。

這段數學推導的核心意義在於:
正相關 (一起漲跌) $\rightarrow$ 壞事。減少下注，砍掉弱勢股。
負相關 (一漲一跌) $\rightarrow$ 好事。增加下注，兩邊都買，資產增長最快。

----
## Taming the Kelly Volatility
我們知道凱利公式賺最快，但它波動太恐怖了，我們該怎麼設計一個『自動煞車系統』來讓資產曲線變平滑？

### 凱利公式太「刺激」了
* 幾何隨機漫步 (Geometric Random Walk)：使用凱利公式投資，資產曲線會像醉漢走路一樣，雖然長期趨勢向上，但中間會劇烈震盪。
* 連續虧損的風險：即使你有優勢（勝率 $p > 0.5$），在有限的時間內，連續輸 $n$ 次的機率 $q^n$ 依然存在。只要時間夠長，災難性的回撤（Catastrophic Drawdown） 遲早會發生
* 時間有限：理論上的「最大增長」需要無限長的時間來實現。但投資人壽命有限，我們更關心「在具體的一段時間後（比如 60 次下注），我能剩下多少錢？」


一般人會用「半凱利（Half-Kelly）」策略，也就是只投凱利建議金額的一半，來降低風險
可以不可用「動態調整（Dynamic Adjustment）」。不固定下注比例，而是根據目前的表現來調整?
* 如果目前賺得比預期多（運氣好） $\rightarrow$ 減少下注比例（落袋為安，守住獲利）
* 如果目前賺得比預期少（運氣差） $\rightarrow$ 增加下注比例（接近凱利最大值，試圖追回平均增長率）


### 雙曲正切函數 ($\tanh$)
#### 為什麼選 $\tanh$？
* S 型曲線：它的形狀像一個 S，可以把任何輸入數值（$x$ 從 $-\infty$ 到 $+\infty$）壓縮到一個固定的區間（$-1$ 到 $1$）
* 平滑過渡：它不會像開關一樣突然跳變，而是平滑地改變數值，適合用來做控制系統
作者把標準的 $\tanh$ 做了位移和縮放，創造了一個調節係數 $y$
$$y = 0.5(1 + tanh((Gt − Ga − h)s))$$
* $G_t$ (Target Growth Rate)：目標增長率（通常設定為凱利公式算出的最大增長率 $G_{max}$）
* $G_a$ (Actual Growth Rate)：實際增長率（你目前真正的績效）
* $G_t - G_a$：這是「落後程度」
	* 如果是正數，代表你落後目標（賺太少），需要增加賭注
	* 如果是負數，代表你超越目標（賺太多），需要減少賭注
* $h$ (Shift)：水平位移參數，用來微調反應的中心點
* $s$ (Scale)：縮放參數，控制曲線的斜率（反應靈敏度）
* $y$：最終算出一個 $0$ 到 $1$ 之間的係數


### 最終決策公式：決定下注比例 $f$
算出調節係數 $y$ 之後，電腦就會根據它來決定這一局該下注多少比例 $f$：$$f = \text{minbet} + y \times (\text{maxbet} - \text{minbet})$$
* minbet：你設定的最小下注比例（保守底線）
* maxbet：你設定的最大下注比例（通常就是凱利值，激進上限）
	* 當 $y$ 接近 0（代表你實際績效 $G_a$ 遠超目標 $G_t$） $\rightarrow$ $f$ 接近 minbet（這局少壓一點，保護獲利）
	* 當 $y$ 接近 1（代表你實際績效 $G_a$ 遠輸目標 $G_t$） $\rightarrow$ $f$ 接近 maxbet（這局壓大一點，回到凱利水準以求追趕）

```
import numpy as np
import matplotlib.pyplot as plt

def run_dynamic_kelly_simulation():
    # --- 1. 參數設定 (完全依照 Algorithm 31 與 Example 7.1) ---
    p = 0.6          # 勝率
    gamma = 1.0      # 賠率 (gain)
    
    # 動態調整參數
    target_growth = 0.02   # Gt: 目標增長率
    min_bet = 0.05         # minbet
    max_bet = 0.20         # maxbet
    
    # Tanh 參數
    h = 0.005
    s = 0.0167

    # 模擬設定
    n_trials = 10000  # 為了驗證書上的 0.0162，我們跑 1萬次
    n_bets = 60       # 書上範例是 60/40 game，通常指 60 次下注，或觀察 60 期
    
    final_growth_rates = [] # 儲存每次模擬的最終增長率
    
    # 為了畫圖，我們隨機選取 3 條路徑存下來
    plot_histories = []
    indices_to_plot = np.random.choice(n_trials, 3, replace=False)

    print("開始模擬 10,000 次 (這可能需要幾秒鐘)...")

    # --- 2. 演算法迴圈 ---
    for i in range(n_trials):
        F = 1.0  # 初始財富
        history = [1.0]
        
        for n in range(1, n_bets + 1):
            # 公式：Ga = (log F) / n
            # 注意：這裡 n 是指「當前是第幾局」。
            # 雖然 n=1 時 log(1)=0，Ga=0，但這是符合書上邏輯的(剛開始沒績效)
            Ga = np.log(F) / n
            
            # 核心公式 (修正了書上的排版錯誤，改用除法)
            # x = Gt - Ga (目標 - 實際)
            x = target_growth - Ga
            
            # y = 0.5 * (1 + tanh((x - h) / s))
            tanh_input = (x - h) / s
            y = 0.5 * (1 + np.tanh(tanh_input))
            
            # 計算下注比例 f
            f = min_bet + y * (max_bet - min_bet)
            
            # 擲硬幣 U ~ U(0,1)
            if np.random.rand() < p:
                F = F * (1 + gamma * f) # Win
            else:
                F = F * (1 - f)         # Lose
            
            history.append(F)

        # 紀錄這一輪的「最終平均增長率」
        # G = (1/n) * log(Fn)
        final_gr = np.log(F) / n_bets
        final_growth_rates.append(final_gr)
        
        # 如果是被選中的路徑，就存起來畫圖
        if i in indices_to_plot:
            plot_histories.append(history)

    # --- 3. 驗證結果 ---
    avg_growth_rate_result = np.mean(final_growth_rates)
    print("-" * 30)
    print(f"書上宣稱的平均增長率: 0.0162")
    print(f"程式模擬的平均增長率: {avg_growth_rate_result:.4f}")
    print("-" * 30)
    if abs(avg_growth_rate_result - 0.0162) < 0.001:
        print(" 驗證成功！結果與書本一致。")
    else:
        print(" 結果有偏差，請檢查參數。")

    return plot_histories, target_growth, n_bets

# 執行模擬
histories, Gt, n_bets = run_dynamic_kelly_simulation()

# --- 4. 畫圖 (完全依照 Fig 7.6 風格) ---
plt.figure(figsize=(10, 6))

# 畫出模擬路徑
for i, h in enumerate(histories):
    plt.plot(h, label=f'Simulation {i+1}', linewidth=1.5, alpha=0.8)

# 畫出目標增長曲線 (Smooth curve)
# F_target = e^(Gt * n)
n_axis = np.arange(n_bets + 1)
target_curve = np.exp(Gt * n_axis)
plt.plot(n_axis, target_curve, 'k--', linewidth=2, label='Target Growth (Smooth)')

# 設定為對數座標 (關鍵)
plt.yscale('log')

plt.title('Dynamic Kelly: Fortune Histories (Log Scale)')
plt.xlabel('Number of Bets (n)')
plt.ylabel('Fortune')
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()
```


## 補充 參數不確定性與「半凱利」 (Half-Kelly)
為什麼要減半？（數學上的「不對稱風險」）
凱利公式的目標是最大化增長率 $G(f)$。如果你把「下注比例 ($f$)」當作 X 軸，「長期增長率 ($G$)」當作 Y 軸畫出來，你會得到一條 拋物線 (Parabola)，就像一座山。
* 山頂 ($f^*$)：這是全凱利 (Full Kelly)。增長率最高，但位於懸崖邊緣
* 左半邊 (Underbetting)：下注太少。雖然賺得慢，但增長率是正的，很安全
* 右半邊 (Overbetting)：下注太多。這是死亡區域

```
import numpy as np
import matplotlib.pyplot as plt

def kelly_growth_rate(f, p, odds):
    """
    計算凱利增長率 G(f)
    f: 下注比例
    p: 勝率
    odds: 賠率 (1賠幾)
    """
    q = 1 - p
    # G(f) = p * ln(1 + odds * f) + q * ln(1 - f)
    # 避免 log(0) 或負數錯誤
    if f >= 1 or f <= -1/odds:
        return -np.inf
    return p * np.log(1 + odds * f) + q * np.log(1 - f)

# --- 設定參數 ---
p_win = 0.6      # 勝率 60%
odds = 1.0       # 賠率 1賠1
f_full = p_win - (1 - p_win) / odds  # 凱利公式 f* = p - q/b = 0.6 - 0.4 = 0.2 (20%)

# 準備繪圖數據
f_values = np.linspace(0, 0.45, 500) # 從 0% 到 45% 下注比例
growth_rates = [kelly_growth_rate(f, p_win, odds) for f in f_values]

# --- 計算關鍵點數值 ---
g_peak = kelly_growth_rate(f_full, p_win, odds)          # 山頂 (Full Kelly)
g_half = kelly_growth_rate(f_full * 0.5, p_win, odds)    # 半山腰 (Half Kelly)
g_double = kelly_growth_rate(f_full * 2.0, p_win, odds)  # 兩倍 (Double Kelly) - 應該接近 0

ratio_return = g_half / g_peak

print(f"--- 數學驗證 ---")
print(f"全凱利 (Full Kelly) 下注: {f_full:.2%} -> 增長率 G: {g_peak:.5f}")
print(f"半凱利 (Half Kelly) 下注: {f_full*0.5:.2%} -> 增長率 G: {g_half:.5f}")
print(f"半凱利的績效比例 (Half/Full): {ratio_return:.2%}")
print(f"半凱利的風險比例: 50% (因為下注金額減半，波動率直接減半)")
print(f"結論: 用 50% 的風險，換到了 {ratio_return:.1%} 的潛在報酬！")


# --- 繪圖 ---
plt.figure(figsize=(12, 10))

# 1. 第一張圖：標準凱利曲線 (證明不對稱性)
plt.subplot(2, 1, 1)
plt.plot(f_values, growth_rates, label='Growth Rate Curve', color='blue', linewidth=2)

# 標記關鍵點
plt.scatter([f_full], [g_peak], color='red', s=100, zorder=5, label=f'Full Kelly ({f_full:.0%})')
plt.scatter([f_full * 0.5], [g_half], color='green', s=100, zorder=5, label=f'Half Kelly ({f_full*0.5:.0%})')
plt.scatter([f_full * 2.0], [g_double], color='black', s=100, zorder=5, label=f'2x Kelly ({f_full*2:.0%})')

# 畫出懸崖區域
plt.axvspan(f_full * 2.0, 0.45, color='red', alpha=0.1, label='Death Zone (G < 0)')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(f_full, color='red', linestyle='--', alpha=0.3)

plt.title(f'Figure 1: The Kelly Parabola (Win Rate={p_win:.0%}, Odds={odds})')
plt.xlabel('Betting Fraction (f)')
plt.ylabel('Expected Growth Rate (G)')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. 第二張圖：預測錯誤的代價 (參數不確定性)
# 模擬情況：你以為勝率是 60%，但實際只有 55%
p_real = 0.55
f_real_optimal = p_real - (1 - p_real) / odds # 真實的最佳解應該是 10%

real_growth_rates = [kelly_growth_rate(f, p_real, odds) for f in f_values]

plt.subplot(2, 1, 2)
# 畫兩條線
plt.plot(f_values, growth_rates, label='Estimated Curve (You THINK p=60%)', color='gray', linestyle='--')
plt.plot(f_values, real_growth_rates, label='REAL Curve (Actually p=55%)', color='red', linewidth=2)

# 標記：如果你用估計的全凱利 (20%) 去下注，落在真實曲線的哪裡？
g_oops_full = kelly_growth_rate(f_full, p_real, odds)
plt.scatter([f_full], [g_oops_full], color='purple', s=150, marker='x', zorder=10, label='Bet Full Kelly (Dangerous!)')

# 標記：如果你用估計的半凱利 (10%) 去下注，落在真實曲線的哪裡？
g_oops_half = kelly_growth_rate(f_full * 0.5, p_real, odds)
plt.scatter([f_full * 0.5], [g_oops_half], color='green', s=150, marker='o', zorder=10, label='Bet Half Kelly (Safe!)')

plt.axvline(f_full * 2.0, color='black', linestyle=':', label='Cliff of Estimation')
plt.title('Figure 2: The Cost of Being Wrong (Estimated 60% vs Real 55%)')
plt.xlabel('Betting Fraction (f)')
plt.ylabel('Real Growth Rate')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

####  數學不對稱性
* 如果你少下一半 ($0.5 f^*$)：你的報酬率大約還有最高點的 75%，但波動率（風險）直接少一半
* 如果你多下一半 ($1.5 f^*$)：你的報酬率會劇烈下降，而且只要你的勝率估算稍微錯一點點，你就會掉進「右邊的懸崖」


我們算出來的勝率（例如 60%）是基於「過去」的數據。如果「未來」的勝率變成 55%，原本算出來的 $f^*$ 就會變成「過度下注」。
為了避免掉進右邊的懸崖（破產），最聰明的做法就是主動往左邊站——也就是只下一般的一半（Half-Kelly）$$f_{real} = 0.5 \times f^*$$

## 連續時間金融與莫頓比例 (Merton Fraction)
書中的凱利公式是用在「丟銅板」這類離散 (Discrete) 事件上的。但股票是連續 (Continuous) 跳動的，我們需要更高級的數學工具：隨機微積分 (Stochastic Calculus)
### 股票的運動模型
我們假設股價 $S$ 遵循「幾何布朗運動 (Geometric Brownian Motion)」：
$$\frac{dS}{S} = \mu dt + \sigma dZ$$
* $\mu$ (Mu)：股票的預期年化報酬率（例如 10%）。這代表股票長期向上的趨勢
* $\sigma$ (Sigma)：股票的年化波動率（標準差，例如 20%）。這代表股票亂跳的程度

### 你的財富增長率公式 (The Growth Rate)
如果你拿出資金的比例 $f$ 去買這支股票，剩下的錢放定存（假設定存利率 $r=0$ 以簡化計算），你的資產 $W$ 的增長率 $G(f)$ 在數學上會變成這樣：$$G(f) = \underbrace{f \mu}_{\text{賺到的報酬}} - \underbrace{\frac{1}{2} f^2 \sigma^2}_{\text{波動拖累 (Volatility Drag)}}$$

為什麼會有後面那一項減號？
* 第一項 $f\mu$：很直覺。你買越多 ($f$ 越大)，或是股票越會漲 ($\mu$ 越高)，你賺越多。這是算術平均。
* 第二項 $-\frac{1}{2} f^2 \sigma^2$：這叫做波動率拖累
	* 還記得我們說過「漲 50% 再跌 50%，結果是虧損 25%」嗎？
	* 在連續時間數學中，這個虧損效應被精確地定義為 $\frac{1}{2}\sigma^2$
	* 因為你有開槓桿 $f$，所以這個拖累會以 平方 ($f^2$) 的速度放大
	* 這解釋了為什麼下注太大 ($f$ 太大)，波動率拖累會吃掉所有的獲利

### 求解最佳 $f$
我們要找到一個 $f$，讓增長率 $G(f)$ 最大
Step 1: 對 $f$ 微分$$\frac{dG}{df} = \frac{d}{df} \left( f\mu - \frac{1}{2} f^2 \sigma^2 \right)$$$$\frac{dG}{df} = \mu - f\sigma^2$$Step 2: 令導數為 0$$\mu - f\sigma^2 = 0$$Step 3: 移項解出 $f$$$f\sigma^2 = \mu$$$$f^* = \frac{\mu}{\sigma^2}$$這就是著名的 莫頓比例 (Merton Fraction)！
(註：如果有無風險利率 $r$，公式會變成 $f^ = \frac{\mu - r}{\sigma^2}$，分子變成超額報酬)

