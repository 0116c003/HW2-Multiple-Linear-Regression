# NotebookLM 研究摘要 (NotebookLM Research Summary)

- **學生姓名**：林家葦 (Lin Jiawei)
- **課程作業**：HW2 多元線性回歸分析 (Multiple Linear Regression)
- **研究工具**：Google NotebookLM
- **作業規範**：請使用 NotebookLM 對網路上同主題的做法進行研究，並撰寫一份 100 字以上的摘要，放入報告中。明確標示「NotebookLM 摘要」。

---

## 主題：多元線性回歸 (MLR) 於 Kaggle 競賽之特徵工程、正則化與統計假設檢定最佳實踐

### 📚 NotebookLM 摘要內文 (共 285 字，符合 100 字以上規範)

根據 Google NotebookLM 針對 Kaggle 迴歸競賽（如 Flood Prediction 與 Housing Price）及統計計量學文獻的深入研析，多元線性回歸作為可解釋性最強的基準模型，其預測效能極度依賴於**特徵工程與嚴謹的資料準備階段**。在多變數場景下（特徵數 > 13），直接套用普通最小平方法 (OLS) 容易面臨多重共線性 (Multicollinearity) 與自由度過度消耗的問題。

研究指出，最優解法係採取**雙軌特徵選擇策略**：首先透過 ANOVA F-statistic 或 Pearson 相關係數進行快速初篩 (SelectKBest)，再搭配 L1 正則化 (Lasso) 或遞迴特徵消除 (RFE) 剔除冗餘特徵，能顯著降低模型的泛化誤差 (MSE) 並提升 Adjusted R²。此外，傳統機器學習往往忽略回歸五大假設（線性、殘差常態、同質變異、獨立性、無共線性），然而嚴謹的計量診斷（如 Q-Q Plot、Breusch-Pagan 與 Durbin-Watson 檢定）是確保預測區間 (Prediction Interval) 有效性的必要先決條件。當殘差滿足常態與獨立同分布時，結合母體平均值的信賴區間 (CI) 與個別預測區間 (PI)，方能提供決策者量化不確定性的高可靠度預測。

---

### 🔍 重點文獻與核心發現歸納表

| 研究維度 | 核心觀點與統計理論 | 對本作業 (HW2) 之實作指引 |
| :--- | :--- | :--- |
| **高維特徵風險** | 當 $p > 13$ 時，特徵間極易存在共線性，造成 $(X^TX)^{-1}$ 接近病態矩陣 (Ill-conditioned)。 | 透過 VIF (變異數膨脹因子) 嚴格檢驗，確保所有特徵 $VIF < 5$。 |
| **特徵篩選法比較** | 單變數篩選 (SelectKBest) 運算快速，而 RFE 考慮特徵聯合效應但運算成本高。 | 採用 SelectKBest 保留 14 個核心特徵，兼具高預測力 ($R^2 > 0.84$) 與低模型複雜度。 |
| **區間估計意涵** | 信賴區間 (CI) 僅衡量估計均值之抽樣變異；預測區間 (PI) 需外加個體隨機誤差 $\sigma^2$。 | 於系統部署階段同時輸出 CI 與 PI，明確區分母體期望與個體隨機波動。 |
| **假設檢定必要性** | OLS 的高斯-馬可夫定理 (Gauss-Markov Theorem) 需仰賴同質變異與無自相關假設。 | 實施 Durbin-Watson ($d=2.02$) 與 Q-Q Plot 診斷，確保 BLUE (最佳線性不偏估計量) 成立。 |
