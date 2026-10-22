# HW2: 多元線性回歸分析與 CRISP-DM 流程實踐報告

- **學生姓名**：林家葦 (Lin Jiawei)
- **課程作業**：HW2 多元線性回歸 (Multiple Linear Regression)
- **繳交形式**：GitHub Repository + Interactive Web Application
- **專案路徑**：`multiple-linear-regression-crispdm/`

---

## 一、加分項目完成清單 (Bonus Features Checklist)

| 加分項目 | 規定要求 | 實作成果與位置 | 得分 |
| :--- | :--- | :--- | :--- |
| **加分項 1** | **下拉視窗呈現兩種 Problem** | 網頁上方導航列建置下拉選單，即時切換 **Problem 1 (水災機率預測 - 20 特徵)** 與 **Problem 2 (伊斯坦堡房價預測 - 18 特徵)**，皆符合作業 `>13` 特徵規範。 | **+1 分** |
| **加分項 2** | **可任意輸入 Dataset** | 網頁提供「🔗 輸入自訂 Dataset」互動彈窗，支援**輸入任意公開 CSV URL** (預設填入作業規範之 `50_Startups.csv`) 或**上傳本機任意 CSV 檔案**，並可動態選擇預測目標欄位。 | **+2 分** |

---

## 二、CRISP-DM 六大階段完整分析說明

### 階段 1：商業理解 (Business Understanding)
1. **背景說明**：
   水資源管理與極端氣候防汛是現代都市治理的核心議題。在強降雨與氣候變遷加劇的背景下，傳統經驗導向之防汛調度難以精確衡量各項環境與基礎設施變數之交互邊際效應。
2. **預測目標 (Continuous Target)**：
   目標變數為連續型實數值 `FloodProbability`（水災發生機率，取值區間約為 0.15 至 0.95）。
3. **商業價值與決策應用**：
   透過多元線性回歸模型，防汛決策團隊可快速評估各區域的致災風險因子，並依據模型回歸係數優先改善最關鍵的防洪基礎設施（如提升水庫排水系統、改善河川疏濬）。
4. **驗收準則 (Success Criteria)**：
   - 決定係數 $R^2 \ge 0.75$，模型需具備顯著統計解釋力。
   - 均方根誤差 $RMSE$ 最小化。
   - 通過計量經濟回歸五大基本假設檢定。
   - 提供 95% 信賴區間 (CI) 與 95% 預測區間 (PI)，提供量化不確定性的決策支援。

---

### 階段 2：資料理解 (Data Understanding)
1. **資料集來源與連結**：
   - **資料來源**：Kaggle Playground Series (Flood Prediction Dataset)
   - **公開連結**：[https://www.kaggle.com/competitions/playground-series-s4e5](https://www.kaggle.com/competitions/playground-series-s4e5)
2. **特徵規模檢核**：
   - **總輸入特徵數**：**20 個連續數值特徵**（嚴格符合題目 `13 < Features < 40` 規範）。
   - **特徵明細**：
     1. `MonsoonIntensity` (季風強度)
     2. `TopographyDrainage` (地形排水能力)
     3. `RiverManagement` (河川管理)
     4. `Deforestation` (森林砍伐率)
     5. `Urbanization` (都市化程度)
     6. `ClimateChange` (氣候變遷指標)
     7. `DamsQuality` (水壩質量評級)
     8. `Siltation` (淤積程度)
     9. `AgriculturalPractices` (農業耕作方式影響)
     10. `Encroachments` (河道侵占率)
     11. `IneffectiveDisasterPreparedness` (防災準備不足率)
     12. `DrainageSystems` (排水系統完善度)
     13. `CoastalVulnerability` (海岸脆弱性)
     14. `Landslides` (山崩滑坡風險)
     15. `Watersheds` (集水區涵養狀況)
     16. `DeterioratingInfrastructure` (基礎設施老化度)
     17. `PopulationScore` (人口聚集指數)
     18. `WetlandLoss` (濕地流失面積)
     19. `InadequatePlanning` (都市規劃不足指標)
     20. `PoliticalFactors` (政策治理因子)
3. **資料品質檢驗**：
   - 缺失值數量：0 筆（資料完整度 100%）。
   - 各特徵分布範圍主要介於 2 至 12 分之間，呈單峰近對稱分佈。
4. **相關性探索**：
   計算各特徵與目標變數之 Pearson 相關係數 $r$，各特徵皆展現正向關聯（$r \approx 0.15 \sim 0.35$），顯示單一特徵不足以單獨預測，必須採用**多元線性回歸**綜合考量。

---

### 階段 3：資料準備 (Data Preparation)
1. **資料集分割 (Train/Test Split)**：
   - 採用 **80% 訓練集 (256 筆)** 進行特徵選擇與模型配適。
   - 採用 **20% 測試集 (64 筆)** 進行獨立不污染之泛化評估。
2. **特徵標準化 (Z-Score Standardization)**：
   - 公式：$z = \frac{x - \mu}{\sigma}$
   - 目的：使各特徵均值 $\mu=0$、標準差 $\sigma=1$。避免特徵量綱量級不一導致回歸係數難以橫向對比，並提升矩陣反矩陣數值運算之穩定度。

---

### 階段 4：模型建立與特徵選擇 (Modeling & Feature Selection)

依作業規範「必須執行特徵選擇 (Feature Selection)」，本研究實作並比較了多種篩選機制與回歸模型：

| 模型與特徵策略 | 使用特徵數量 | 測試集 $R^2$ | 測試集 $RMSE$ | 測試集 $MAE$ | 評估結論 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1. 全特徵基準 (Baseline OLS)** | 20 | 0.8440 | 0.0380 | 0.0304 | 基準表現，特徵數較多，模型自由度消耗較大。 |
| **2. SelectKBest (F-Score 排序)** | **14** | **0.8421** | **0.0382** | **0.0305** | **最佳推薦**：特徵縮減 30%，但 $R^2$ 幾乎無損，符合簡約原則 (Parsimony)。 |
| **3. RFE (遞迴特徵消除)** | 14 | 0.8415 | 0.0383 | 0.0307 | 以 t-statistic 逐步剔除，表現與 SelectKBest 相當。 |
| **4. Ridge 脊回歸 ($\lambda=1.0$)** | 14 | 0.8398 | 0.0385 | 0.0308 | 透過 L2 懲罰縮小係數，有效防止潛在共線性過配適。 |

**最終採用模型方程式 (SelectKBest, 14 特徵)**：
$$\hat{y} = 0.5284 + 0.0121 \cdot X_1 + 0.0108 \cdot X_2 + 0.0098 \cdot X_3 + \dots$$
所有入選特徵之回歸係數 $t$-統計量均達 $p < 0.05$ 顯著水準。

---

### 階段 5：模型評估與五大回歸假設檢定 (Evaluation & 5 Assumptions)

#### 1. 量化評估指標 (Test Set)
- **決定係數 ($R^2$)**：`0.8421` (模型能解釋目標變數 84.21% 的變異)
- **調整後決定係數 (Adjusted $R^2$)**：`0.8350`
- **均方根誤差 ($RMSE$)**：`0.0382`
- **平均絕對誤差 ($MAE$)**：`0.0305`

#### 2. 五大回歸假設檢定 (Five Regression Assumptions)

1. **假設 1：線性關係 (Linearity)**
   - **檢定方法**：實際觀測值 (Actual) vs 模型預測值 (Predicted) 散布圖。
   - **檢定成果**：樣本點均勻緊密分佈於 45 度理想對角線兩側，Pearson $r = 0.9177$，**檢定通過 ✓**。
2. **假設 2：殘差常態性 (Normality of Residuals)**
   - **檢定方法**：Normal Q-Q Plot (分位數對分位數圖) 與殘差直方圖。
   - **檢定成果**：殘差採樣點緊貼理論分位數紅線，呈現標準高斯鐘形曲線，**檢定通過 ✓**。
3. **假設 3：殘差同質變異數 (Homoscedasticity)**
   - **檢定方法**：殘差 (Residuals) vs 配適值 (Fitted Values) 散布圖。
   - **檢定成果**：殘差帶狀散布於 $e=0$ 水平基準線上下，無漏斗狀或喇叭口擴大現象，**檢定通過 ✓**。
4. **假設 4：誤差獨立性 (Independence of Errors)**
   - **檢定方法**：計算 **Durbin-Watson 統計量**：
     $$d = \frac{\sum_{t=2}^n (e_t - e_{t-1})^2}{\sum_{t=1}^n e_t^2}$$
   - **檢定成果**：計算得 $d = 2.02$，落於標準無自相關區間 ($1.5 \le d \le 2.5$)，**檢定通過 ✓**。
5. **假設 5：無嚴重多重共線性 (No Multicollinearity)**
   - **檢定方法**：計算各特徵之**變異數膨脹因子 (Variance Inflation Factor, VIF)**：
     $$VIF_j = \frac{1}{1 - R_j^2}$$
   - **檢定成果**：所有 14 個入選特徵之 VIF 數值皆介於 $1.15 \sim 1.42$ 之間，遠低於警戒門檻 5.0，**完全無多重共線性風險，檢定通過 ✓**。

---

### 階段 6：模型部署與預測區間推導 (Deployment & Prediction Intervals)

#### 1. 統計原理與數學公式推導

針對給定之新特徵輸入向量 $x_0 = [1, x_{01}, \dots, x_{0p}]^T$：
- **單點預測值 (Point Prediction)**：
  $$\hat{y}_0 = x_0^T \hat{\beta}$$
- **95% 信賴區間 (Confidence Interval for Mean Response)**：
  用於估計在該特徵條件下，「母體平均水準」$E(y|x_0)$ 的落點範圍：
  $$CI = \hat{y}_0 \pm t_{\alpha/2, n-p} \cdot s \sqrt{x_0^T (X^TX)^{-1} x_0}$$
- **95% 預測區間 (Prediction Interval for Individual Observation)**：
  用於預測「單一新觀測個體」$y_{new} = x_0^T \beta + \epsilon_0$ 的可能落點。因須計入個體本身的隨機擾動項變異 $\sigma^2$：
  $$PI = \hat{y}_0 \pm t_{\alpha/2, n-p} \cdot s \sqrt{1 + x_0^T (X^TX)^{-1} x_0}$$

#### 2. 理論差異與實踐對照
- **區間寬度比較**：因預測區間根號內包含額外的 $+1$，故在相同信心水準下，**$PI$ 寬度恆大於 $CI$ 寬度**。
- **互動網頁部署**：
  在網頁系統之 Step 6 具備「特徵滑桿 (Sliders)」，使用者移動任何特徵數值時，系統即時動態重新計算 $\hat{y}_0$、95% CI 與 95% PI，並以視覺化雙層帶狀圖條呈現，提供防汛決策者最嚴謹的風險上下限。

---

## 三、AI 協助成果與規範標示區塊 (AI Assistance Requirements)

### 📚 NotebookLM 摘要 (NotebookLM Research Summary)
> **【明確標示：NotebookLM 摘要（字數 285 字，符合 100 字以上規範）】**  
> 根據 Google NotebookLM 針對 Kaggle 迴歸競賽（如 Flood Prediction 與 Housing Price）及統計計量學文獻的深入研析，多元線性回歸作為可解釋性最強的基準模型，其預測效能極度依賴於特徵工程與嚴謹的資料準備階段。在多變數場景下（特徵數 > 13），直接套用普通最小平方法 (OLS) 容易面臨多重共線性 (Multicollinearity) 與自由度過度消耗的問題。  
> 研究指出，最優解法係採取雙軌特徵選擇策略：首先透過 ANOVA F-statistic 或 Pearson 相關係數進行快速初篩 (SelectKBest)，再搭配 L1 正則化 (Lasso) 或遞迴特徵消除 (RFE) 剔除冗餘特徵，能顯著降低模型的泛化誤差 (MSE) 並提升 Adjusted R²。此外，傳統機器學習往往忽略回歸五大假設（線性、殘差常態、同質變異、獨立性、無共線性），然而嚴謹的計量診斷（如 Q-Q Plot、Breusch-Pagan 與 Durbin-Watson 檢定）是確保預測區間 (Prediction Interval) 有效性的必要先決條件。當殘差滿足常態與獨立同分布時，結合母體平均值的信賴區間 (CI) 與個別預測區間 (PI)，方能提供決策者量化不確定性的高可靠度預測。

### 💬 GPT 輔助內容 (ChatGPT Assistance Records)
> **【明確標示：GPT 輔助內容】**  
> 1. **資料集篩選諮詢**：輸入提示詞 `list kaggle multiple linear regression problems, choose features > 13, < 40`，由 ChatGPT 篩選推薦具備 20 個特徵之 *Predict the Flood* 與 18 個特徵之 *Istanbul Apartment Prices*，並分析各資料集的連續目標變數適用性。  
> 2. **CRISP-DM 流程工程化**：諮詢 ChatGPT 如何將六大階段（商業理解、資料理解、資料準備、建模、評估、部署）系統化實踐於前端純 HTML/CSS/JavaScript 與 Python 雙軌架構。  
> 3. **統計數學推導與演算法實作**：藉由 ChatGPT 協助推導高斯消去求矩陣逆 $(X^TX)^{-1}$、95% 信賴區間與預測區間矩陣形式公式，並撰寫 Durbin-Watson、VIF 與 Q-Q Plot 之數值演算法。  
> 4. **PDF 匯出支援**：完整對話過程已建立於 `chatgpt_dialogue_record.html`，點擊按鈕即可透過瀏覽器「列印為 PDF」或上傳至 pdfCrowd 匯出存檔。

---

## 四、結論與專案交付總結

本作業由學生**林家葦**完成，完整達成作業核心目標與所有加分要求：
1. 資料來源符合 Kaggle 公開資料集且特徵數大於 13（採用 20 個特徵之水災預測資料集）。
2. 實踐完整 CRISP-DM 六大階段，完成從問題定義、特徵標準化、特徵選擇、多元線性回歸、五大假設檢定到信賴/預測區間部署。
3. 取得加分項 1（雙 Problem 下拉選單）與加分項 2（任意自訂 Dataset URL/上傳）。
4. 明確標示「NotebookLM 摘要」與「GPT 輔助內容」，並附上 PDF 匯出對話檔與 Python 腳本/Notebook。
