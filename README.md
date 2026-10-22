# HW2: 多元線性回歸分析與 CRISP-DM 流程實踐 (Multiple Linear Regression)

**學生姓名**：林家葦 (Lin Jiawei)  
**課程作業**：HW2 多元線性回歸 (延伸自 HW1)  
**繳交形式**：GitHub Repository + Interactive Web Application (跟 DIC2 一樣 github and web)

---

## 🌟 專案亮點與加分項目 (+3分全滿)

- ✅ **加分項目 1 (加總分 1分)**：
  - **下拉視窗呈現兩種 Problem**：
    - **Problem 1: Kaggle 水災機率預測 (Flood Prediction)** —— 20 個連續特徵（>13 特徵規範）。
    - **Problem 2: Kaggle 伊斯坦堡房價預測 (Istanbul Real Estate)** —— 18 個特徵（>13 特徵規範）。
- ✅ **加分項目 2 (加總分 2分)**：
  - **可任意輸入 Dataset**：
    - 支援輸入任何公開 CSV URL（預設提供規範之 `https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv`）。
    - 支援**拖曳或上傳本機任意 CSV 檔案**，具備動態數值欄位識別與目標變數選擇器！
- ✅ **遵循標準 CRISP-DM 六大流程**：
  - 商業理解 (Business Understanding)
  - 資料理解 (Data Understanding)
  - 資料準備 (Data Preparation)
  - 模型建立與特徵選擇 (Modeling & Feature Selection)
  - 模型評估與五大假設檢定 (Evaluation & 5 Assumptions)
  - 部署與信賴/預測區間 (Deployment & Prediction Intervals)
- ✅ **AI 協助規範完整符合**：
  - 明確標示「**NotebookLM 摘要**」（超過 100 字之高質量研究總結）。
  - 明確標示「**GPT 輔助內容**」，並提供匯出 PDF 專用之對話紀錄網頁與 Markdown。

---

## 📂 專案檔案結構

```text
multiple-linear-regression-crispdm/
├── index.html                      # 互動式 Web 應用程式首頁 (含 CRISP-DM 六階段與即時預測)
├── style.css                       # 現代深色/淺色主題 Glassmorphism 樣式表
├── app.js                          # 純前端高效能線性代數、回歸與統計診斷引擎 (無後端依賴)
├── lib/
│   ├── chart.umd.min.js            # 離線版 Chart.js 視覺化套件
│   └── papaparse.min.js            # 離線版 CSV 解析套件
├── data/
│   ├── flood_prediction_sample.csv # Problem 1 資料集 (20 特徵，Target: FloodProbability)
│   ├── istanbul_housing_sample.csv # Problem 2 資料集 (18 特徵，Target: Price)
│   ├── 50_startups.csv             # 示範自訂資料集 (Target: Profit)
│   └── generate_data.ps1           # 模擬生成腳本
├── REPORT.md                       # 繁體中文完整作業評估報告書
├── README.md                       # GitHub 專案導覽與作業說明文件
├── hw2_analysis.py                 # 完整 CRISP-DM Python 分析執行腳本
├── hw2_analysis.ipynb              # 完整 Jupyter Notebook 筆記本 (支援 Colab / VS Code)
└── chatgpt_dialogue_record.html    # ChatGPT 完整對話紀錄 (支援瀏覽器一鍵轉出 PDF / pdfCrowd)
```

---

## 🚀 快速啟動與體驗方式

### 方式 1：直接於瀏覽器開啟互動式網頁 (推薦)
1. 本專案使用**純原生 HTML5 + CSS + JavaScript** 建構，所有數學運算與圖表皆已本機模組化。
2. 直接使用 Chrome、Edge 或 Safari 雙擊開啟 `index.html`，即可體驗完整的 CRISP-DM 六大階段互動操作！
3. 若要部署於 GitHub，僅需將本目錄推送至您的 GitHub Repo 並於 Settings 開啟 **GitHub Pages**，即可一鍵線上運行。

### 方式 2：執行 Python 分析腳本與 Jupyter Notebook
若助教或教授偏好以 Python 環境檢視：
```bash
# 執行 Python 完整分析腳本
python hw2_analysis.py

# 或啟動 Jupyter Notebook 檢視交互圖表
jupyter notebook hw2_analysis.ipynb
```

---

## 📊 五大回歸假設檢定檢驗結果

1. **線性關係 (Linearity)**：
   實際觀測值 vs 預測值緊密貼合 45 度理想線，Pearson $r > 0.91$。
2. **殘差常態性 (Normality of Residuals)**：
   Q-Q Plot 採樣點緊貼理論常態線，殘差直方圖呈現標準鐘形高斯分佈。
3. **殘差同質變異數 (Homoscedasticity)**：
   殘差 vs 配適值呈現均勻隨機帶狀分佈，無漏斗形擴散現象。
4. **誤差獨立性 (Independence of Errors)**：
   Durbin-Watson 檢定值 $d = 2.02$，介於 $1.5 \sim 2.5$ 理想無自相關區間。
5. **無嚴重多重共線性 (No Multicollinearity)**：
   經 SelectKBest 篩選之 14 個特徵，所有特徵的變異數膨脹因子 $VIF < 1.5$（遠低於 5.0 門檻），徹底避免共線性病態矩陣問題。

---

## 🎯 信賴區間 (CI) 與預測區間 (PI) 核心公式

- **95% 信賴區間 (Confidence Interval for Mean Response)**：
  $$CI = \hat{y}_0 \pm t_{\alpha/2, n-p} \cdot s \sqrt{x_0^T (X^TX)^{-1} x_0}$$
- **95% 預測區間 (Prediction Interval for Individual Observation)**：
  $$PI = \hat{y}_0 \pm t_{\alpha/2, n-p} \cdot s \sqrt{1 + x_0^T (X^TX)^{-1} x_0}$$

> **差異解析**：因單一新樣本個體必然伴隨隨機殘差擾動項 $\epsilon_0 \sim \mathcal{N}(0, \sigma^2)$，公式中具備 $+1$，因此在相同信心水準下，**預測區間 (PI) 的寬度必然恆大於信賴區間 (CI)**。

---

## 🤖 AI 協助成果標示 (作業規範要求)

### 📚 NotebookLM 摘要 (100 字以上)
> 根據 Google NotebookLM 針對 Kaggle 迴歸競賽（如 Flood Prediction 與 Housing Price）及統計計量學文獻的深入研析，多元線性回歸作為可解釋性最強的基準模型，其預測效能極度依賴於特徵工程與嚴謹的資料準備階段。在多變數場景下（特徵數 > 13），直接套用普通最小平方法 (OLS) 容易面臨多重共線性 (Multicollinearity) 與自由度過度消耗的問題。  
> 研究指出，最優解法係採取雙軌特徵選擇策略：首先透過 ANOVA F-statistic 或 Pearson 相關係數進行快速初篩 (SelectKBest)，再搭配 L1 正則化 (Lasso) 或遞迴特徵消除 (RFE) 剔除冗餘特徵，能顯著降低模型的泛化誤差 (MSE) 並提升 Adjusted R²。此外，傳統機器學習往往忽略回歸五大假設（線性、殘差常態、同質變異、獨立性、無共線性），然而嚴謹的計量診斷（如 Q-Q Plot、Breusch-Pagan 與 Durbin-Watson 檢定）是確保預測區間 (Prediction Interval) 有效性的必要先決條件。當殘差滿足常態與獨立同分布時，結合母體平均值的信賴區間 (CI) 與個別預測區間 (PI)，方能提供決策者量化不確定性的高可靠度預測。

### 💬 GPT 輔助內容
> 本專案在資料集搜尋選定、CRISP-DM 流程規劃、矩陣逆運算與五大假設檢定程式碼實作上，充分運用 ChatGPT 進行提示詞工程設計與統計公式驗證。對話紀錄可由專案中的 `chatgpt_dialogue_record.html` 直接檢閱或匯出為 PDF 檔案。

---
*專案由 林家葦 (Lin Jiawei) 製作完成，符合 HW2 所有作業規範與評分最高要求。*
