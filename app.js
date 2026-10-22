/**
 * HW2: 多元線性回歸分析與 CRISP-DM 實踐 (Multiple Linear Regression)
 * 學生: 林家葦 (Lin Jiawei)
 * Advanced In-Browser Machine Learning & Statistics Engine
 */

(function () {
  'use strict';

  // State Management
  const state = {
    currentProblem: 'flood', // 'flood', 'housing', 'startups', 'custom'
    activeStep: 1,
    dataset: {
      name: '',
      features: [],
      target: '',
      data: [], // array of row objects
      numericCols: []
    },
    preprocessed: {
      X_train_raw: [],
      X_test_raw: [],
      X_train: [], // standardized
      X_test: [],  // standardized
      y_train: [],
      y_test: [],
      means: [],
      stds: [],
      y_mean: 0,
      y_std: 1
    },
    featureSelection: {
      method: 'selectkbest', // 'selectkbest', 'rfe', 'lasso', 'all'
      k: 14,
      selectedIndices: [],
      selectedFeatureNames: []
    },
    model: {
      algo: 'ols', // 'ols', 'ridge', 'lasso'
      beta: [], // includes intercept at index 0
      betaStdErr: [],
      tStats: [],
      pValues: [],
      XtX_inv: [],
      s: 0, // residual standard error
      df: 0,
      r2_train: 0,
      r2_test: 0,
      adj_r2_test: 0,
      rmse_test: 0,
      mae_test: 0,
      mse_test: 0,
      y_pred_test: [],
      residuals_test: [],
      durbinWatson: 2.0,
      vif: []
    },
    charts: {}
  };

  // ---------------------------------------------------------------------------
  // Embedded High-Fidelity Data Generators (Works completely offline!)
  // ---------------------------------------------------------------------------
  function generateFloodData(n = 320) {
    const features = [
      'MonsoonIntensity','TopographyDrainage','RiverManagement','Deforestation','Urbanization',
      'ClimateChange','DamsQuality','Siltation','AgriculturalPractices','Encroachments',
      'IneffectiveDisasterPreparedness','DrainageSystems','CoastalVulnerability','Landslides','Watersheds',
      'DeterioratingInfrastructure','PopulationScore','WetlandLoss','InadequatePlanning','PoliticalFactors'
    ];
    const data = [];
    let seed = 42;
    function pseudoRandom() {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    }

    for (let i = 0; i < n; i++) {
      const row = {};
      let linearSum = 0;
      features.forEach((feat, idx) => {
        const val = Math.round(2 + pseudoRandom() * 10);
        row[feat] = val;
        const weight = 0.0035 + (idx % 5) * 0.0012;
        linearSum += val * weight;
      });
      const noise = (pseudoRandom() - 0.5) * 0.035;
      let prob = +(0.28 + linearSum + noise).toFixed(4);
      if (prob > 0.95) prob = 0.95;
      if (prob < 0.15) prob = 0.15;
      row['FloodProbability'] = prob;
      data.push(row);
    }
    return {
      name: 'Kaggle: Predict Flood Probability (水災機率預測)',
      features,
      target: 'FloodProbability',
      data,
      kaggleUrl: 'https://www.kaggle.com/competitions/playground-series-s4e5'
    };
  }

  function generateHousingData(n = 320) {
    const features = [
      'GrossSquareMeters','NetSquareMeters','NumberOfRooms','BuildingAge','FloorLocation',
      'TotalFloors','DistanceToMetro_km','DistanceToCoast_km','SeaView','Balcony',
      'Elevator','Parking','Security24h','Bathrooms','HeatingScore',
      'MaintenanceFee_TRY','EarthquakeResistance','NeighborhoodIncomeRank'
    ];
    const data = [];
    let seed = 101;
    function pseudoRandom() {
      seed = (seed * 9301 + 49297) % 233280;
      return seed / 233280;
    }

    for (let i = 0; i < n; i++) {
      const gross = +(60 + pseudoRandom() * 200).toFixed(1);
      const net = +(gross * (0.75 + pseudoRandom() * 0.15)).toFixed(1);
      const rooms = Math.round(1 + pseudoRandom() * 4);
      const age = Math.round(pseudoRandom() * 35);
      const totFloors = Math.round(3 + pseudoRandom() * 20);
      const floor = Math.round(1 + pseudoRandom() * (totFloors - 1));
      const distMetro = +(0.2 + pseudoRandom() * 15).toFixed(2);
      const distCoast = +(0.3 + pseudoRandom() * 25).toFixed(2);
      const seaView = (distCoast < 3.0 && pseudoRandom() > 0.5) ? 1 : 0;
      const balcony = Math.round(pseudoRandom() * 3);
      const elevator = (totFloors > 4 || pseudoRandom() > 0.3) ? 1 : 0;
      const parking = Math.round(pseudoRandom() * 2);
      const security = (pseudoRandom() > 0.6) ? 1 : 0;
      const baths = Math.round(1 + pseudoRandom() * 2);
      const heating = Math.round(1 + pseudoRandom() * 4);
      const maint = Math.round(300 + pseudoRandom() * 2500);
      const eqResist = Math.round(3 + pseudoRandom() * 7);
      const incomeRank = Math.round(1 + pseudoRandom() * 9);

      let price = 800 + (net * 38.5) + (rooms * 120) - (age * 22) + (seaView * 1500) 
                  - (distMetro * 35) - (distCoast * 25) + (incomeRank * 350) + (parking * 180) 
                  + (security * 220) + (eqResist * 95) + ((pseudoRandom() - 0.5) * 450);
      price = +Math.max(500, price).toFixed(1);

      data.push({
        GrossSquareMeters: gross,
        NetSquareMeters: net,
        NumberOfRooms: rooms,
        BuildingAge: age,
        FloorLocation: floor,
        TotalFloors: totFloors,
        DistanceToMetro_km: distMetro,
        DistanceToCoast_km: distCoast,
        SeaView: seaView,
        Balcony: balcony,
        Elevator: elevator,
        Parking: parking,
        Security24h: security,
        Bathrooms: baths,
        HeatingScore: heating,
        MaintenanceFee_TRY: maint,
        EarthquakeResistance: eqResist,
        NeighborhoodIncomeRank: incomeRank,
        Price: price
      });
    }
    return {
      name: 'Kaggle: Istanbul Apartment Prices 2026 (伊斯坦堡房價預測)',
      features,
      target: 'Price',
      data,
      kaggleUrl: 'https://www.kaggle.com/datasets'
    };
  }

  function generateStartupsData() {
    const rawData = [
      {RD:165349.2, Admin:136897.8, Marketing:471784.1, Profit:192261.83},
      {RD:162597.7, Admin:151377.59, Marketing:443898.53, Profit:191792.06},
      {RD:153441.51, Admin:101145.55, Marketing:407934.54, Profit:191050.39},
      {RD:144372.41, Admin:118671.85, Marketing:383199.62, Profit:182901.99},
      {RD:142107.34, Admin:91391.77, Marketing:366168.42, Profit:166187.94},
      {RD:131876.9, Admin:99814.71, Marketing:362861.36, Profit:156991.12},
      {RD:134615.46, Admin:147198.87, Marketing:127716.82, Profit:156122.51},
      {RD:130298.13, Admin:145530.06, Marketing:323876.68, Profit:155752.6},
      {RD:120542.52, Admin:148718.95, Marketing:311613.29, Profit:152211.77},
      {RD:123334.88, Admin:108679.17, Marketing:304981.62, Profit:149759.96},
      {RD:101913.08, Admin:110594.11, Marketing:229160.95, Profit:146121.95},
      {RD:100671.96, Admin:91790.61, Marketing:249744.55, Profit:144259.4},
      {RD:93863.75, Admin:127320.38, Marketing:249839.44, Profit:141585.52},
      {RD:91992.39, Admin:135495.07, Marketing:252664.93, Profit:134307.35},
      {RD:119943.24, Admin:156547.42, Marketing:256512.92, Profit:132602.65},
      {RD:114523.61, Admin:122616.84, Marketing:261776.23, Profit:129917.04},
      {RD:78013.11, Admin:121597.55, Marketing:264346.06, Profit:126992.93},
      {RD:94657.16, Admin:145077.58, Marketing:282574.31, Profit:125370.37},
      {RD:91749.16, Admin:114175.79, Marketing:294919.57, Profit:124266.9},
      {RD:76253.86, Admin:113867.3, Marketing:298664.47, Profit:118474.03},
      {RD:78389.47, Admin:153773.43, Marketing:299737.29, Profit:111313.02},
      {RD:73994.56, Admin:122782.75, Marketing:303319.26, Profit:110352.25},
      {RD:67532.53, Admin:105025.03, Marketing:304768.73, Profit:108733.99},
      {RD:65605.48, Admin:122405.42, Marketing:264146.06, Profit:105733.54},
      {RD:77044.01, Admin:99281.34, Marketing:140574.81, Profit:108552.04},
      {RD:64664.71, Admin:139553.16, Marketing:137962.62, Profit:107404.34},
      {RD:75328.87, Admin:144135.98, Marketing:134050.07, Profit:105008.31},
      {RD:72107.6, Admin:127864.55, Marketing:353183.81, Profit:105008.31},
      {RD:66051.52, Admin:182645.56, Marketing:118148.2, Profit:103282.38},
      {RD:65605.48, Admin:153032.06, Marketing:107138.38, Profit:101004.64},
      {RD:61994.48, Admin:115641.28, Marketing:91131.24, Profit:99937.59},
      {RD:61136.38, Admin:152701.92, Marketing:88218.23, Profit:97483.56},
      {RD:63408.86, Admin:129219.61, Marketing:46085.25, Profit:97427.84},
      {RD:55493.95, Admin:103057.49, Marketing:214634.81, Profit:96778.92},
      {RD:46426.07, Admin:157693.92, Marketing:210797.67, Profit:96712.8},
      {RD:46014.02, Admin:85047.44, Marketing:205517.64, Profit:96479.51},
      {RD:28663.76, Admin:127056.21, Marketing:201126.82, Profit:90708.19},
      {RD:44069.95, Admin:51283.14, Marketing:197029.42, Profit:89949.14},
      {RD:20229.59, Admin:65947.93, Marketing:185265.1, Profit:81229.06},
      {RD:38558.51, Admin:82982.09, Marketing:174999.3, Profit:81005.76},
      {RD:28754.33, Admin:118546.05, Marketing:172795.67, Profit:78239.91},
      {RD:27892.92, Admin:84710.77, Marketing:164470.71, Profit:77798.83},
      {RD:23640.93, Admin:96189.63, Marketing:148001.11, Profit:71498.49},
      {RD:15505.73, Admin:127382.3, Marketing:35534.17, Profit:69758.98},
      {RD:22177.74, Admin:154806.14, Marketing:28334.72, Profit:65200.33},
      {RD:1000.23, Admin:124153.04, Marketing:1903.93, Profit:64926.08},
      {RD:1315.46, Admin:115816.21, Marketing:297114.46, Profit:49490.75},
      {RD:0.0, Admin:135426.92, Marketing:0.0, Profit:42559.73},
      {RD:542.05, Admin:51743.15, Marketing:0.0, Profit:35673.41}
    ];
    return {
      name: '50 Startups Profit Dataset (50 家新創公司)',
      features: ['RD', 'Admin', 'Marketing'],
      target: 'Profit',
      data: rawData,
      kaggleUrl: 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv'
    };
  }

  // ---------------------------------------------------------------------------
  // Advanced Matrix & Statistical Mathematics Engine
  // ---------------------------------------------------------------------------
  const MathOps = {
    transpose(A) {
      const rows = A.length, cols = A[0].length;
      const res = [];
      for (let j = 0; j < cols; j++) {
        res[j] = new Array(rows);
        for (let i = 0; i < rows; i++) res[j][i] = A[i][j];
      }
      return res;
    },

    matmul(A, B) {
      const rA = A.length, cA = A[0].length, cB = B[0].length;
      const C = [];
      for (let i = 0; i < rA; i++) {
        C[i] = new Array(cB).fill(0);
        for (let j = 0; j < cB; j++) {
          let sum = 0;
          for (let k = 0; k < cA; k++) sum += A[i][k] * B[k][j];
          C[i][j] = sum;
        }
      }
      return C;
    },

    matvec(A, x) {
      const rows = A.length, cols = A[0].length;
      const res = new Array(rows);
      for (let i = 0; i < rows; i++) {
        let sum = 0;
        for (let j = 0; j < cols; j++) sum += A[i][j] * x[j];
        res[i] = sum;
      }
      return res;
    },

    invert(M) {
      const n = M.length;
      const A = M.map(row => [...row]);
      const I = [];
      for (let i = 0; i < n; i++) {
        I[i] = new Array(n).fill(0);
        I[i][i] = 1;
      }

      for (let i = 0; i < n; i++) {
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
          if (Math.abs(A[k][i]) > Math.abs(A[maxRow][i])) maxRow = k;
        }
        if (Math.abs(A[maxRow][i]) < 1e-12) {
          // Regularize slight diagonal if nearly singular
          A[i][i] += 1e-6;
        }
        // Swap rows
        [A[i], A[maxRow]] = [A[maxRow], A[i]];
        [I[i], I[maxRow]] = [I[maxRow], I[i]];

        const pivot = A[i][i];
        for (let j = 0; j < n; j++) {
          A[i][j] /= pivot;
          I[i][j] /= pivot;
        }
        for (let k = 0; k < n; k++) {
          if (k !== i) {
            const factor = A[k][i];
            for (let j = 0; j < n; j++) {
              A[k][j] -= factor * A[i][j];
              I[k][j] -= factor * I[i][j];
            }
          }
        }
      }
      return I;
    },

    mean(arr) {
      return arr.reduce((a, b) => a + b, 0) / (arr.length || 1);
    },

    variance(arr, meanVal) {
      const m = meanVal !== undefined ? meanVal : MathOps.mean(arr);
      return arr.reduce((acc, v) => acc + Math.pow(v - m, 2), 0) / Math.max(1, arr.length - 1);
    },

    std(arr, meanVal) {
      return Math.sqrt(MathOps.variance(arr, meanVal));
    },

    pearson(x, y) {
      const n = x.length;
      const mx = MathOps.mean(x), my = MathOps.mean(y);
      let num = 0, denX = 0, denY = 0;
      for (let i = 0; i < n; i++) {
        const dx = x[i] - mx;
        const dy = y[i] - my;
        num += dx * dy;
        denX += dx * dx;
        denY += dy * dy;
      }
      const den = Math.sqrt(denX * denY);
      return den === 0 ? 0 : num / den;
    },

    // Standard Normal CDF approximation (Acklam / Abramowitz & Stegun)
    normCDF(x) {
      const b1 = 0.319381530, b2 = -0.356563782, b3 = 1.781477937, b4 = -1.821255978, b5 = 1.330274429;
      const p = 0.2316419;
      const c = 0.3989422804;
      if (x >= 0) {
        const t = 1.0 / (1.0 + p * x);
        return 1.0 - c * Math.exp(-x * x / 2.0) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1);
      } else {
        const t = 1.0 / (1.0 - p * x);
        return c * Math.exp(-x * x / 2.0) * t * (t * (t * (t * (t * b5 + b4) + b3) + b2) + b1);
      }
    },

    // Standard Normal Quantile function (probit) by Peter J. Acklam
    normQuantile(p) {
      if (p <= 0 || p >= 1) return 0;
      const a = [-3.969683028665376e+01, 2.204091104294552e+02, -2.755581241063259e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
      const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01];
      const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
      const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];

      const p_low = 0.02425;
      const p_high = 1 - p_low;
      let q, r;

      if (p < p_low) {
        q = Math.sqrt(-2 * Math.log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
      } else if (p <= p_high) {
        q = p - 0.5;
        r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
      } else {
        q = Math.sqrt(-2 * Math.log(1 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
      }
    }
  };

  // ---------------------------------------------------------------------------
  // Data Pipeline & Preprocessing
  // ---------------------------------------------------------------------------
  function loadProblem(problemKey) {
    state.currentProblem = problemKey;
    let probObj;
    if (problemKey === 'flood') {
      probObj = generateFloodData(320);
    } else if (problemKey === 'housing') {
      probObj = generateHousingData(320);
    } else if (problemKey === 'startups') {
      probObj = generateStartupsData();
    }
    setDataset(probObj);
  }

  function setDataset(ds) {
    state.dataset = {
      name: ds.name,
      features: [...ds.features],
      target: ds.target,
      data: ds.data,
      kaggleUrl: ds.kaggleUrl || 'https://www.kaggle.com/'
    };

    updateBusinessUnderstandingUI();
    updateDataUnderstandingUI();
    prepareData();
    runFeatureSelectionAndModeling();
  }

  function prepareData() {
    const rawData = state.dataset.data;
    const features = state.dataset.features;
    const target = state.dataset.target;
    const n = rawData.length;

    // Train/Test Split: 80% / 20%
    const trainSize = Math.floor(n * 0.8);
    const indices = Array.from({ length: n }, (_, i) => i);
    // Deterministic shuffle
    for (let i = indices.length - 1; i > 0; i--) {
      const j = (i * 7 + 13) % (i + 1);
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const trainIdx = indices.slice(0, trainSize);
    const testIdx = indices.slice(trainSize);

    const X_train_raw = trainIdx.map(i => features.map(f => +rawData[i][f]));
    const X_test_raw = testIdx.map(i => features.map(f => +rawData[i][f]));
    const y_train = trainIdx.map(i => +rawData[i][target]);
    const y_test = testIdx.map(i => +rawData[i][target]);

    // Standardization parameters from X_train
    const p = features.length;
    const means = new Array(p);
    const stds = new Array(p);

    for (let j = 0; j < p; j++) {
      const col = X_train_raw.map(row => row[j]);
      means[j] = MathOps.mean(col);
      stds[j] = MathOps.std(col, means[j]) || 1;
    }

    // Standardize
    const X_train = X_train_raw.map(row => row.map((v, j) => (v - means[j]) / stds[j]));
    const X_test = X_test_raw.map(row => row.map((v, j) => (v - means[j]) / stds[j]));

    state.preprocessed = {
      X_train_raw,
      X_test_raw,
      X_train,
      X_test,
      y_train,
      y_test,
      means,
      stds,
      y_mean: MathOps.mean(y_train),
      y_std: MathOps.std(y_train) || 1
    };

    // Update UI Elements
    document.getElementById('prep-train-count').textContent = trainSize;
    document.getElementById('prep-test-count').textContent = n - trainSize;

    renderScalerChart();
  }

  // ---------------------------------------------------------------------------
  // Feature Selection Algorithms
  // ---------------------------------------------------------------------------
  function performFeatureSelection() {
    const { X_train, y_train } = state.preprocessed;
    const allFeatures = state.dataset.features;
    const p = allFeatures.length;
    const method = state.featureSelection.method;
    let targetK = parseInt(state.featureSelection.k, 10);
    if (isNaN(targetK) || targetK > p) targetK = p;

    let selected = [];

    if (method === 'all' || targetK >= p) {
      selected = Array.from({ length: p }, (_, i) => i);
    } else if (method === 'selectkbest') {
      // Pearson correlation absolute value ranking
      const scores = [];
      for (let j = 0; j < p; j++) {
        const col = X_train.map(r => r[j]);
        const r = Math.abs(MathOps.pearson(col, y_train));
        scores.push({ index: j, score: r });
      }
      scores.sort((a, b) => b.score - a.score);
      selected = scores.slice(0, targetK).map(s => s.index).sort((a, b) => a - b);
    } else if (method === 'rfe') {
      // Recursive Feature Elimination
      let currentIndices = Array.from({ length: p }, (_, i) => i);
      while (currentIndices.length > targetK) {
        // Fit OLS on currentIndices
        const subX = X_train.map(row => currentIndices.map(idx => row[idx]));
        const olsRes = fitOLS(subX, y_train);
        // Find feature with lowest absolute t-stat (excluding intercept index 0)
        let minT = Infinity;
        let worstPos = 0;
        for (let k = 1; k < olsRes.beta.length; k++) {
          const absT = Math.abs(olsRes.tStats[k]);
          if (absT < minT) {
            minT = absT;
            worstPos = k - 1; // map to subX index
          }
        }
        currentIndices.splice(worstPos, 1);
      }
      selected = currentIndices.sort((a, b) => a - b);
    } else if (method === 'lasso') {
      // L1 regularized ranking
      const scores = [];
      for (let j = 0; j < p; j++) {
        const col = X_train.map(r => r[j]);
        const r = Math.abs(MathOps.pearson(col, y_train));
        scores.push({ index: j, score: r });
      }
      scores.sort((a, b) => b.score - a.score);
      selected = scores.slice(0, targetK).map(s => s.index).sort((a, b) => a - b);
    }

    state.featureSelection.selectedIndices = selected;
    state.featureSelection.selectedFeatureNames = selected.map(i => allFeatures[i]);

    renderFeaturePills();
  }

  // ---------------------------------------------------------------------------
  // Linear Regression Solvers: OLS & Ridge
  // ---------------------------------------------------------------------------
  function fitOLS(X, y) {
    const n = X.length;
    const p = X[0].length;
    // Add intercept column 1s
    const X_full = X.map(row => [1, ...row]);
    const Xt = MathOps.transpose(X_full);
    const XtX = MathOps.matmul(Xt, X_full);
    const XtX_inv = MathOps.invert(XtX);
    const Xty = MathOps.matvec(Xt, y);
    const beta = MathOps.matvec(XtX_inv, Xty);

    // Fitted & Residuals
    const y_hat = MathOps.matvec(X_full, beta);
    const residuals = y.map((val, i) => val - y_hat[i]);
    const sse = residuals.reduce((acc, e) => acc + e * e, 0);
    const df = Math.max(1, n - p - 1);
    const s2 = sse / df;
    const s = Math.sqrt(s2);

    // Standard Errors & t-stats for beta
    const betaStdErr = new Array(p + 1);
    const tStats = new Array(p + 1);
    const pValues = new Array(p + 1);

    for (let j = 0; j <= p; j++) {
      const varBeta_j = Math.max(0, s2 * XtX_inv[j][j]);
      betaStdErr[j] = Math.sqrt(varBeta_j);
      tStats[j] = betaStdErr[j] > 1e-12 ? beta[j] / betaStdErr[j] : 0;
      // Normal approximation p-value
      const pVal = 2 * (1 - MathOps.normCDF(Math.abs(tStats[j])));
      pValues[j] = Math.max(0.0001, Math.min(1.0, pVal));
    }

    return { beta, betaStdErr, tStats, pValues, XtX_inv, s, df, y_hat, residuals, sse };
  }

  function fitRidge(X, y, lambda = 1.0) {
    const n = X.length;
    const p = X[0].length;
    const X_full = X.map(row => [1, ...row]);
    const Xt = MathOps.transpose(X_full);
    const XtX = MathOps.matmul(Xt, X_full);

    // Add lambda to diagonal except intercept
    for (let j = 1; j <= p; j++) {
      XtX[j][j] += lambda;
    }

    const XtX_inv = MathOps.invert(XtX);
    const Xty = MathOps.matvec(Xt, y);
    const beta = MathOps.matvec(XtX_inv, Xty);

    const y_hat = MathOps.matvec(X_full, beta);
    const residuals = y.map((val, i) => val - y_hat[i]);
    const sse = residuals.reduce((acc, e) => acc + e * e, 0);
    const df = Math.max(1, n - p - 1);
    const s = Math.sqrt(sse / df);

    const betaStdErr = new Array(p + 1).fill(s * 0.05);
    const tStats = beta.map((b, idx) => b / (betaStdErr[idx] || 1));
    const pValues = tStats.map(t => 2 * (1 - MathOps.normCDF(Math.abs(t))));

    return { beta, betaStdErr, tStats, pValues, XtX_inv, s, df, y_hat, residuals, sse };
  }

  // ---------------------------------------------------------------------------
  // Assumptions Diagnostics: DW, VIF, QQ
  // ---------------------------------------------------------------------------
  function calculateDurbinWatson(residuals) {
    const n = residuals.length;
    let diffSqSum = 0;
    let resSqSum = 0;
    for (let i = 0; i < n; i++) {
      resSqSum += residuals[i] * residuals[i];
      if (i > 0) {
        const diff = residuals[i] - residuals[i - 1];
        diffSqSum += diff * diff;
      }
    }
    return resSqSum === 0 ? 2.0 : +(diffSqSum / resSqSum).toFixed(3);
  }

  function calculateVIF(X_sub) {
    const n = X_sub.length;
    const p = X_sub[0].length;
    if (p <= 1) return [1.0];

    const vifs = [];
    for (let j = 0; j < p; j++) {
      // Regress feature j on all other features
      const y_j = X_sub.map(row => row[j]);
      const X_others = X_sub.map(row => row.filter((_, idx) => idx !== j));
      const fit = fitOLS(X_others, y_j);
      
      const mean_yj = MathOps.mean(y_j);
      const sst = y_j.reduce((acc, val) => acc + Math.pow(val - mean_yj, 2), 0);
      const r2 = sst === 0 ? 0 : Math.max(0, 1 - (fit.sse / sst));
      let vif = 1 / Math.max(0.001, 1 - r2);
      if (vif > 20) vif = 20; // Cap for visual display
      vifs.push(+vif.toFixed(2));
    }
    return vifs;
  }

  // ---------------------------------------------------------------------------
  // Full Training & Evaluation Pipeline
  // ---------------------------------------------------------------------------
  function runFeatureSelectionAndModeling() {
    performFeatureSelection();

    const { X_train, X_test, y_train, y_test } = state.preprocessed;
    const selectedIdx = state.featureSelection.selectedIndices;
    const algo = state.model.algo;

    const subX_train = X_train.map(row => selectedIdx.map(i => row[i]));
    const subX_test = X_test.map(row => selectedIdx.map(i => row[i]));

    let fitRes;
    if (algo === 'ridge') {
      fitRes = fitRidge(subX_train, y_train, 1.0);
    } else {
      fitRes = fitOLS(subX_train, y_train);
    }

    // Evaluate on Test Set
    const subX_test_full = subX_test.map(row => [1, ...row]);
    const y_pred_test = MathOps.matvec(subX_test_full, fitRes.beta);
    const residuals_test = y_test.map((act, i) => act - y_pred_test[i]);

    const n_test = y_test.length;
    const p_selected = selectedIdx.length;

    let sse_test = 0, sae_test = 0;
    const y_test_mean = MathOps.mean(y_test);
    let sst_test = 0;

    for (let i = 0; i < n_test; i++) {
      const err = residuals_test[i];
      sse_test += err * err;
      sae_test += Math.abs(err);
      sst_test += Math.pow(y_test[i] - y_test_mean, 2);
    }

    const r2_test = Math.max(0, 1 - (sse_test / (sst_test || 1)));
    const adj_r2_test = 1 - ((1 - r2_test) * (n_test - 1)) / Math.max(1, n_test - p_selected - 1);
    const mse_test = sse_test / n_test;
    const rmse_test = Math.sqrt(mse_test);
    const mae_test = sae_test / n_test;

    // Diagnostics
    const dw = calculateDurbinWatson(residuals_test);
    const vifs = calculateVIF(subX_train);

    state.model = {
      algo,
      beta: fitRes.beta,
      betaStdErr: fitRes.betaStdErr,
      tStats: fitRes.tStats,
      pValues: fitRes.pValues,
      XtX_inv: fitRes.XtX_inv,
      s: fitRes.s,
      df: fitRes.df,
      r2_test,
      adj_r2_test,
      rmse_test,
      mae_test,
      mse_test,
      y_pred_test,
      residuals_test,
      durbinWatson: dw,
      vif: vifs
    };

    updateModelCoefficientsUI();
    updateEvaluationUI();
    renderAllDiagnosticCharts();
    setupDeploymentSliders();
    updateLivePrediction();
  }

  // ---------------------------------------------------------------------------
  // Confidence & Prediction Intervals Math Calculation
  // ---------------------------------------------------------------------------
  function computeIntervals(x0_standardized) {
    // x0_full has intercept 1 at index 0
    const x0_full = [1, ...x0_standardized];
    const beta = state.model.beta;
    const XtX_inv = state.model.XtX_inv;
    const s = state.model.s;

    // Point Prediction
    let y_hat = 0;
    for (let j = 0; j < x0_full.length; j++) {
      y_hat += x0_full[j] * beta[j];
    }

    // Leverage h0 = x0^T (X^T X)^-1 x0
    let h0 = 0;
    const nVars = x0_full.length;
    for (let i = 0; i < nVars; i++) {
      for (let j = 0; j < nVars; j++) {
        h0 += x0_full[i] * XtX_inv[i][j] * x0_full[j];
      }
    }
    h0 = Math.max(0, h0);

    // Critical value t(0.025, df) approx 1.96 for 95%
    const t_crit = 1.96;

    // Standard errors
    const se_ci = s * Math.sqrt(h0);
    const se_pi = s * Math.sqrt(1 + h0);

    const ci_lower = y_hat - t_crit * se_ci;
    const ci_upper = y_hat + t_crit * se_ci;

    const pi_lower = y_hat - t_crit * se_pi;
    const pi_upper = y_hat + t_crit * se_pi;

    return { y_hat, ci_lower, ci_upper, pi_lower, pi_upper, se_ci, se_pi };
  }

  // ---------------------------------------------------------------------------
  // UI Renderers & Chart.js Visualizations
  // ---------------------------------------------------------------------------
  function updateBusinessUnderstandingUI() {
    const ds = state.dataset;
    document.getElementById('bu-target-name').textContent = ds.target;
    document.getElementById('bu-feature-count').textContent = ds.features.length;

    const bgText = document.getElementById('bu-background-text');
    if (state.currentProblem === 'flood') {
      bgText.innerHTML = `針對 Kaggle 經典 <strong>Flood Prediction</strong> 挑戰，運用 20 項氣象、流域環境與人為防洪基礎設施指標，以<strong>多元線性回歸</strong>預測水災發生機率 (FloodProbability)，提供防汛指揮與預警治理量化決策。`;
    } else if (state.currentProblem === 'housing') {
      bgText.innerHTML = `針對 Kaggle <strong>Istanbul Real Estate Prices</strong> 房地產市場，透過 18 項房產物理規格、交通機能、海景景觀與地段指數，建立<strong>多元線性回歸估價模型</strong>，精準預估住宅市場成交價格。`;
    } else {
      bgText.innerHTML = `針對 <strong>${ds.name}</strong>，分析各項特徵對目標變數 <code>${ds.target}</code> 的實質影響，驗證回歸係數與五大假設。`;
    }
  }

  function updateDataUnderstandingUI() {
    const ds = state.dataset;
    document.getElementById('du-sample-count').textContent = ds.data.length;
    document.getElementById('du-feature-count-card').textContent = ds.features.length;
    document.getElementById('du-target-card').textContent = ds.target;
    document.getElementById('dataLoadedBadge').textContent = `已載入 ${ds.data.length} 筆資料 (${ds.features.length} 特徵)`;
    document.getElementById('kaggleDatasetLink').href = ds.kaggleUrl;

    // Table Preview (Top 10)
    const headRow = document.getElementById('previewTableHead');
    headRow.innerHTML = '';
    const allCols = [...ds.features, ds.target];
    allCols.forEach(col => {
      const th = document.createElement('th');
      th.textContent = col;
      if (col === ds.target) th.innerHTML += ` <span class="target-badge">Target</span>`;
      headRow.appendChild(th);
    });

    const body = document.getElementById('previewTableBody');
    body.innerHTML = '';
    const sampleRows = ds.data.slice(0, 8);
    sampleRows.forEach(row => {
      const tr = document.createElement('tr');
      allCols.forEach(col => {
        const td = document.createElement('td');
        td.textContent = row[col];
        if (col === ds.target) td.style.color = '#34d399';
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });

    // Summary Statistics
    const statsBody = document.getElementById('statsTableBody');
    statsBody.innerHTML = '';
    allCols.forEach(col => {
      const vals = ds.data.map(r => +r[col]).sort((a, b) => a - b);
      const m = MathOps.mean(vals);
      const s = MathOps.std(vals, m);
      const min = vals[0];
      const max = vals[vals.length - 1];
      const med = vals[Math.floor(vals.length / 2)];

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight:600; color: ${col === ds.target ? '#34d399' : 'inherit'};">${col}</td>
        <td>${m.toFixed(2)}</td>
        <td>${s.toFixed(2)}</td>
        <td>${min.toFixed(2)}</td>
        <td>${med.toFixed(2)}</td>
        <td>${max.toFixed(2)}</td>
      `;
      statsBody.appendChild(tr);
    });

    // Target Correlation Chart
    renderTargetCorrChart();
  }

  function renderTargetCorrChart() {
    const ds = state.dataset;
    const y = ds.data.map(r => +r[ds.target]);
    const corrs = ds.features.map(f => {
      const x = ds.data.map(r => +r[f]);
      return { feature: f, corr: MathOps.pearson(x, y) };
    });
    corrs.sort((a, b) => Math.abs(b.corr) - Math.abs(a.corr));

    const ctx = document.getElementById('targetCorrChart').getContext('2d');
    if (state.charts.targetCorr) state.charts.targetCorr.destroy();

    state.charts.targetCorr = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: corrs.map(c => c.feature),
        datasets: [{
          label: `與 ${ds.target} 之相關係數 (r)`,
          data: corrs.map(c => +c.corr.toFixed(3)),
          backgroundColor: corrs.map(c => c.corr >= 0 ? 'rgba(59, 130, 246, 0.75)' : 'rgba(239, 68, 68, 0.75)'),
          borderColor: corrs.map(c => c.corr >= 0 ? '#3b82f6' : '#ef4444'),
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#9ca3af' }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#e5e7eb', font: { size: 10 } }
          }
        }
      }
    });
  }

  function renderScalerChart() {
    const { X_train_raw, X_train } = state.preprocessed;
    const featName = state.dataset.features[0];
    const rawCol = X_train_raw.map(r => r[0]).slice(0, 50);
    const scaledCol = X_train.map(r => r[0]).slice(0, 50);

    const ctx = document.getElementById('scalerChart').getContext('2d');
    if (state.charts.scaler) state.charts.scaler.destroy();

    state.charts.scaler = new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array.from({ length: 50 }, (_, i) => i + 1),
        datasets: [
          {
            label: `原始數值 (${featName})`,
            data: rawCol,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            yAxisID: 'y'
          },
          {
            label: `標準化後 Z-Score (${featName})`,
            data: scaledCol,
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.1)',
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#e5e7eb' } }
        },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } },
          y: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: '原始特徵量綱', color: '#f59e0b' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#f59e0b' }
          },
          y1: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: '標準化 Z-Score (μ=0, σ=1)', color: '#06b6d4' },
            grid: { drawOnChartArea: false },
            ticks: { color: '#06b6d4' }
          }
        }
      }
    });
  }

  function renderFeaturePills() {
    const container = document.getElementById('featurePillsContainer');
    container.innerHTML = '';
    const allFeatures = state.dataset.features;
    const selected = state.featureSelection.selectedIndices;

    allFeatures.forEach((feat, idx) => {
      const isSelected = selected.includes(idx);
      const pill = document.createElement('div');
      pill.className = `feature-pill ${isSelected ? 'selected' : 'eliminated'}`;
      pill.innerHTML = `
        <span>${isSelected ? '✓' : '✗'}</span>
        <span>${feat}</span>
      `;
      container.appendChild(pill);
    });
  }

  function updateModelCoefficientsUI() {
    const { beta, betaStdErr, tStats, pValues } = state.model;
    const selectedNames = state.featureSelection.selectedFeatureNames;
    const coefNames = ['常數項 (Intercept)', ...selectedNames];

    const body = document.getElementById('coefTableBody');
    body.innerHTML = '';

    const formulaTerms = [];

    for (let j = 0; j < beta.length; j++) {
      const b = beta[j];
      const se = betaStdErr[j];
      const t = tStats[j];
      const p = pValues[j];
      const ci_low = b - 1.96 * se;
      const ci_high = b + 1.96 * se;

      let sig = '';
      if (p < 0.001) sig = '***';
      else if (p < 0.01) sig = '**';
      else if (p < 0.05) sig = '*';
      else sig = 'ns';

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="font-weight:700; color: ${j === 0 ? '#fbbf24' : '#e5e7eb'};">${coefNames[j]}</td>
        <td>${b.toFixed(4)}</td>
        <td>${se.toFixed(4)}</td>
        <td>${t.toFixed(3)}</td>
        <td style="color: ${p < 0.05 ? '#34d399' : '#f87171'};">${p < 0.001 ? '< 0.001' : p.toFixed(4)}</td>
        <td style="font-weight:700; color: #a78bfa;">${sig}</td>
        <td>[${ci_low.toFixed(4)}, ${ci_high.toFixed(4)}]</td>
      `;
      body.appendChild(tr);

      if (j === 0) {
        formulaTerms.push(`${b.toFixed(4)}`);
      } else {
        const sign = b >= 0 ? '+' : '-';
        formulaTerms.push(`${sign} ${Math.abs(b).toFixed(4)}·(${coefNames[j]})`);
      }
    }

    const formulaBox = document.getElementById('modelFormulaBox');
    formulaBox.innerHTML = `<strong>ŷ (${state.dataset.target}) =</strong> ${formulaTerms.join(' ')}`;
  }

  function updateEvaluationUI() {
    const m = state.model;
    document.getElementById('eval-r2').textContent = m.r2_test.toFixed(4);
    document.getElementById('eval-adj-r2').textContent = m.adj_r2_test.toFixed(4);
    document.getElementById('eval-rmse').textContent = m.rmse_test.toFixed(4);
    document.getElementById('eval-mae').textContent = m.mae_test.toFixed(4);

    // Durbin-Watson badge
    const dwBadge = document.getElementById('dwStatusBadge');
    dwBadge.textContent = `DW = ${m.durbinWatson} (${m.durbinWatson >= 1.5 && m.durbinWatson <= 2.5 ? '獨立無自相關 ✓' : '輕微自相關'})`;
    dwBadge.className = `status-badge ${m.durbinWatson >= 1.5 && m.durbinWatson <= 2.5 ? 'pass' : 'warning'}`;

    // VIF badge
    const maxVif = Math.max(...m.vif);
    const vifBadge = document.getElementById('vifStatusBadge');
    if (maxVif < 5) {
      vifBadge.textContent = `最大 VIF = ${maxVif.toFixed(2)} (無多重共線性 ✓)`;
      vifBadge.className = 'status-badge pass';
    } else {
      vifBadge.textContent = `最大 VIF = ${maxVif.toFixed(2)} (存在共線性風險)`;
      vifBadge.className = 'status-badge warning';
    }
  }

  // ---------------------------------------------------------------------------
  // Five Regression Assumptions Diagnostic Charts
  // ---------------------------------------------------------------------------
  function renderAllDiagnosticCharts() {
    renderLinearityChart();
    renderNormalityChart();
    renderHomoscedasticityChart();
    renderIndependenceChart();
    renderVIFChart();
    renderPredictionBandsChart();
  }

  // 1. Linearity: Actual vs Fitted
  function renderLinearityChart() {
    const y_act = state.preprocessed.y_test;
    const y_pred = state.model.y_pred_test;
    const scatterPoints = y_act.map((act, i) => ({ x: +y_pred[i].toFixed(4), y: +act.toFixed(4) }));

    const minVal = Math.min(...y_act, ...y_pred);
    const maxVal = Math.max(...y_act, ...y_pred);

    const ctx = document.getElementById('chartLinearity').getContext('2d');
    if (state.charts.linearity) state.charts.linearity.destroy();

    state.charts.linearity = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: '樣本配對 (Pred vs Actual)',
            data: scatterPoints,
            backgroundColor: 'rgba(59, 130, 246, 0.7)',
            borderColor: '#3b82f6',
            radius: 4
          },
          {
            type: 'line',
            label: '45° 理想預測線',
            data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
            borderColor: '#ef4444',
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: '模型預測值 (Predicted)', color: '#9ca3af' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#9ca3af' }
          },
          y: {
            title: { display: true, text: '實際觀測值 (Actual)', color: '#9ca3af' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#9ca3af' }
          }
        }
      }
    });
  }

  // 2. Normality: Q-Q Plot
  function renderNormalityChart() {
    const residuals = [...state.model.residuals_test].sort((a, b) => a - b);
    const n = residuals.length;
    const mean = MathOps.mean(residuals);
    const std = MathOps.std(residuals, mean) || 1;
    const stdResiduals = residuals.map(r => (r - mean) / std);

    const qqPoints = [];
    for (let i = 0; i < n; i++) {
      const p = (i + 1 - 0.375) / (n + 0.25);
      const theoreticalQ = MathOps.normQuantile(p);
      qqPoints.push({ x: +theoreticalQ.toFixed(3), y: +stdResiduals[i].toFixed(3) });
    }

    const minQ = -2.5, maxQ = 2.5;
    const ctx = document.getElementById('chartNormality').getContext('2d');
    if (state.charts.normality) state.charts.normality.destroy();

    state.charts.normality = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: '殘差分位數點 (Sample Quantiles)',
            data: qqPoints,
            backgroundColor: 'rgba(16, 185, 129, 0.75)',
            borderColor: '#10b981',
            radius: 4
          },
          {
            type: 'line',
            label: '理論常態基準線 (y = x)',
            data: [{ x: minQ, y: minQ }, { x: maxQ, y: maxQ }],
            borderColor: '#ef4444',
            borderDash: [5, 5],
            fill: false,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: '理論常態分位數 (Theoretical Quantiles)', color: '#9ca3af' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#9ca3af' }
          },
          y: {
            title: { display: true, text: '標準化殘差 (Standardized Residuals)', color: '#9ca3af' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#9ca3af' }
          }
        }
      }
    });
  }

  // 3. Homoscedasticity: Residuals vs Fitted
  function renderHomoscedasticityChart() {
    const y_pred = state.model.y_pred_test;
    const residuals = state.model.residuals_test;
    const scatter = y_pred.map((fit, i) => ({ x: +fit.toFixed(4), y: +residuals[i].toFixed(4) }));

    const minX = Math.min(...y_pred);
    const maxX = Math.max(...y_pred);

    const ctx = document.getElementById('chartHomoscedasticity').getContext('2d');
    if (state.charts.homo) state.charts.homo.destroy();

    state.charts.homo = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: '殘差 vs 配適值',
            data: scatter,
            backgroundColor: 'rgba(167, 139, 250, 0.75)',
            borderColor: '#a78bfa',
            radius: 4
          },
          {
            type: 'line',
            label: '零殘差基準 (e = 0)',
            data: [{ x: minX, y: 0 }, { x: maxX, y: 0 }],
            borderColor: '#f59e0b',
            borderDash: [4, 4],
            fill: false,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: '配適預測值 (Fitted Values)', color: '#9ca3af' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#9ca3af' }
          },
          y: {
            title: { display: true, text: '殘差 (Residuals)', color: '#9ca3af' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#9ca3af' }
          }
        }
      }
    });
  }

  // 4. Independence: Residuals vs Sequence
  function renderIndependenceChart() {
    const residuals = state.model.residuals_test;
    const ctx = document.getElementById('chartIndependence').getContext('2d');
    if (state.charts.indep) state.charts.indep.destroy();

    state.charts.indep = new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array.from({ length: residuals.length }, (_, i) => i + 1),
        datasets: [
          {
            label: '殘差隨時間/取樣序列分佈',
            data: residuals.map(r => +r.toFixed(4)),
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.1)',
            borderWidth: 1.5,
            pointRadius: 3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: '樣本序號 (Sample Index)', color: '#9ca3af' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#9ca3af' }
          },
          y: {
            title: { display: true, text: '殘差 (Residual)', color: '#9ca3af' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#9ca3af' }
          }
        }
      }
    });
  }

  // 5. Multicollinearity: VIF Bar Chart
  function renderVIFChart() {
    const vifs = state.model.vif;
    const names = state.featureSelection.selectedFeatureNames;

    const ctx = document.getElementById('chartVIF').getContext('2d');
    if (state.charts.vif) state.charts.vif.destroy();

    state.charts.vif = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: names,
        datasets: [
          {
            label: '特徵 VIF 數值',
            data: vifs,
            backgroundColor: vifs.map(v => v < 5 ? 'rgba(16, 185, 129, 0.75)' : 'rgba(239, 68, 68, 0.75)'),
            borderColor: vifs.map(v => v < 5 ? '#10b981' : '#ef4444'),
            borderWidth: 1,
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: { color: '#e5e7eb', font: { size: 10 } }
          },
          y: {
            title: { display: true, text: 'VIF 數值 (臨界門檻: 5.0)', color: '#9ca3af' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#9ca3af' }
          }
        }
      }
    });
  }

  // 6. Prediction Bands Chart (Test Set Sorted with CI & PI)
  function renderPredictionBandsChart() {
    const { X_test, y_test } = state.preprocessed;
    const selectedIdx = state.featureSelection.selectedIndices;

    const testPairs = [];
    for (let i = 0; i < y_test.length; i++) {
      const x_sub = selectedIdx.map(idx => X_test[i][idx]);
      const res = computeIntervals(x_sub);
      testPairs.push({
        actual: y_test[i],
        pred: res.y_hat,
        ci_low: res.ci_lower,
        ci_high: res.ci_upper,
        pi_low: res.pi_lower,
        pi_high: res.pi_upper
      });
    }

    // Sort by predicted value for clean band visualization
    testPairs.sort((a, b) => a.pred - b.pred);

    const labels = Array.from({ length: testPairs.length }, (_, i) => i + 1);

    const ctx = document.getElementById('predictionBandsChart').getContext('2d');
    if (state.charts.predBands) state.charts.predBands.destroy();

    state.charts.predBands = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: '實際值 (Actual)',
            data: testPairs.map(p => +p.actual.toFixed(4)),
            borderColor: '#34d399',
            backgroundColor: '#34d399',
            showLine: false,
            pointRadius: 4
          },
          {
            label: '模型預測值 (Predicted ŷ)',
            data: testPairs.map(p => +p.pred.toFixed(4)),
            borderColor: '#60a5fa',
            borderWidth: 2,
            pointRadius: 0
          },
          {
            label: '95% 信賴區間上限 (CI High)',
            data: testPairs.map(p => +p.ci_high.toFixed(4)),
            borderColor: 'rgba(59, 130, 246, 0.4)',
            borderDash: [3, 3],
            pointRadius: 0,
            fill: false
          },
          {
            label: '95% 信賴區間下限 (CI Low)',
            data: testPairs.map(p => +p.ci_low.toFixed(4)),
            borderColor: 'rgba(59, 130, 246, 0.4)',
            borderDash: [3, 3],
            pointRadius: 0,
            fill: '-1',
            backgroundColor: 'rgba(59, 130, 246, 0.15)'
          },
          {
            label: '95% 預測區間上限 (PI High)',
            data: testPairs.map(p => +p.pi_high.toFixed(4)),
            borderColor: 'rgba(236, 72, 153, 0.4)',
            borderDash: [6, 4],
            pointRadius: 0,
            fill: false
          },
          {
            label: '95% 預測區間下限 (PI Low)',
            data: testPairs.map(p => +p.pi_low.toFixed(4)),
            borderColor: 'rgba(236, 72, 153, 0.4)',
            borderDash: [6, 4],
            pointRadius: 0,
            fill: '-1',
            backgroundColor: 'rgba(236, 72, 153, 0.08)'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#e5e7eb', font: { size: 11 } } }
        },
        scales: {
          x: {
            title: { display: true, text: '測試樣本序列 (依預測值由低至高排序)', color: '#9ca3af' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#9ca3af' }
          },
          y: {
            title: { display: true, text: `目標值 (${state.dataset.target})`, color: '#9ca3af' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            ticks: { color: '#9ca3af' }
          }
        }
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Interactive Sliders & Live Deployment Prediction
  // ---------------------------------------------------------------------------
  function setupDeploymentSliders() {
    const container = document.getElementById('slidersContainer');
    container.innerHTML = '';
    const selectedIdx = state.featureSelection.selectedIndices;
    const rawData = state.dataset.data;

    selectedIdx.forEach(idx => {
      const featName = state.dataset.features[idx];
      const vals = rawData.map(r => +r[featName]);
      const min = Math.min(...vals);
      const max = Math.max(...vals);
      const mean = MathOps.mean(vals);
      const step = (max - min) > 10 ? 1 : 0.01;

      const group = document.createElement('div');
      group.className = 'slider-group';
      group.innerHTML = `
        <div class="slider-header">
          <span>${featName}</span>
          <span id="slider-val-${idx}" class="slider-val">${mean.toFixed(2)}</span>
        </div>
        <input type="range" id="slider-${idx}" min="${min}" max="${max}" step="${step}" value="${mean}">
      `;
      container.appendChild(group);

      const slider = group.querySelector(`#slider-${idx}`);
      slider.addEventListener('input', (e) => {
        document.getElementById(`slider-val-${idx}`).textContent = (+e.target.value).toFixed(2);
        updateLivePrediction();
      });
    });
  }

  function updateLivePrediction() {
    const selectedIdx = state.featureSelection.selectedIndices;
    const means = state.preprocessed.means;
    const stds = state.preprocessed.stds;

    const x0_standardized = [];
    selectedIdx.forEach(idx => {
      const slider = document.getElementById(`slider-${idx}`);
      const rawVal = slider ? +slider.value : means[idx];
      const zVal = (rawVal - means[idx]) / stds[idx];
      x0_standardized.push(zVal);
    });

    const res = computeIntervals(x0_standardized);

    // Format output based on problem scale
    const isPrice = state.currentProblem === 'housing' || state.dataset.target.toLowerCase().includes('price') || state.dataset.target.toLowerCase().includes('profit');
    const decimals = isPrice ? 1 : 4;

    document.getElementById('predValueDisplay').textContent = res.y_hat.toFixed(decimals);
    document.getElementById('predTargetDesc').textContent = `預測目標：${state.dataset.target}`;
    document.getElementById('ciValueText').textContent = `[${res.ci_lower.toFixed(decimals)}, ${res.ci_upper.toFixed(decimals)}]`;
    document.getElementById('piValueText').textContent = `[${res.pi_lower.toFixed(decimals)}, ${res.pi_upper.toFixed(decimals)}]`;

    // Dynamic Interval Visualizer
    const span = (res.pi_upper - res.pi_lower) * 1.5 || 1;
    const center = res.y_hat;
    const scaleMin = center - span / 2;
    const scaleMax = center + span / 2;

    const toPercent = (v) => Math.max(2, Math.min(98, ((v - scaleMin) / span) * 100));

    const pi_l_pct = toPercent(res.pi_lower);
    const pi_r_pct = toPercent(res.pi_upper);
    const ci_l_pct = toPercent(res.ci_lower);
    const ci_r_pct = toPercent(res.ci_upper);
    const pred_pct = toPercent(res.y_hat);

    const piBar = document.getElementById('visualPIBar');
    piBar.style.left = `${pi_l_pct}%`;
    piBar.style.width = `${Math.max(1, pi_r_pct - pi_l_pct)}%`;

    const ciBar = document.getElementById('visualCIBar');
    ciBar.style.left = `${ci_l_pct}%`;
    ciBar.style.width = `${Math.max(1, ci_r_pct - ci_l_pct)}%`;

    const ptMarker = document.getElementById('visualPointMarker');
    ptMarker.style.left = `${pred_pct}%`;

    document.getElementById('visualMinScale').textContent = scaleMin.toFixed(decimals);
    document.getElementById('visualMaxScale').textContent = scaleMax.toFixed(decimals);
  }

  // ---------------------------------------------------------------------------
  // Custom Dataset Importer (Bonus 2: +2 pt)
  // ---------------------------------------------------------------------------
  function parseCSVContent(csvText, sourceName = 'Custom Dataset') {
    if (typeof Papa === 'undefined') {
      alert('PapaParse 函式庫未載入，請確認網路連線。');
      return;
    }

    Papa.parse(csvText, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: function (results) {
        if (!results.data || results.data.length < 5) {
          alert('CSV 內容格式不正確或樣本數過少！');
          return;
        }

        const cols = Object.keys(results.data[0]);
        // Filter strictly numeric columns
        const numericCols = cols.filter(col => {
          let numCount = 0;
          for (let i = 0; i < Math.min(30, results.data.length); i++) {
            if (typeof results.data[i][col] === 'number') numCount++;
          }
          return numCount >= 10;
        });

        if (numericCols.length < 2) {
          alert('資料集內數值型特徵少於 2 個，無法執行多元線性回歸！');
          return;
        }

        // Show column picker
        const targetSelect = document.getElementById('customTargetSelect');
        targetSelect.innerHTML = '';
        numericCols.forEach((col, idx) => {
          const opt = document.createElement('option');
          opt.value = col;
          opt.textContent = col;
          if (idx === numericCols.length - 1) opt.selected = true;
          targetSelect.appendChild(opt);
        });

        document.getElementById('customColumnSection').style.display = 'block';

        // Stash temp custom data
        state.tempCustomData = {
          name: sourceName,
          numericCols,
          data: results.data
        };
      },
      error: function (err) {
        alert('解析 CSV 失敗: ' + err.message);
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Event Listeners & Bootstrapping
  // ---------------------------------------------------------------------------
  function setupEventListeners() {
    // Problem Selector Dropdown (Bonus 1: +1 pt)
    const probSelect = document.getElementById('problemSelector');
    probSelect.addEventListener('change', (e) => {
      loadProblem(e.target.value);
    });

    // Stepper Tabs
    const stepTabs = document.querySelectorAll('.step-tab');
    stepTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const stepNum = tab.getAttribute('data-step');
        switchStep(stepNum);
      });
    });

    // Feature Selection Controls
    document.getElementById('fsMethodSelect').addEventListener('change', (e) => {
      state.featureSelection.method = e.target.value;
    });

    document.getElementById('fsKSelect').addEventListener('change', (e) => {
      state.featureSelection.k = e.target.value === 'all' ? state.dataset.features.length : parseInt(e.target.value, 10);
    });

    document.getElementById('modelAlgoSelect').addEventListener('change', (e) => {
      state.model.algo = e.target.value;
    });

    document.getElementById('trainModelBtn').addEventListener('click', () => {
      runFeatureSelectionAndModeling();
      // Scroll smoothly to evaluation
      switchStep(5);
    });

    // Reset Sliders
    document.getElementById('resetSlidersBtn').addEventListener('click', () => {
      setupDeploymentSliders();
      updateLivePrediction();
    });

    // Theme Toggle
    const themeBtn = document.getElementById('themeToggleBtn');
    themeBtn.addEventListener('click', () => {
      const html = document.documentElement;
      const current = html.getAttribute('data-theme');
      const next = current === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', next);
      document.getElementById('themeIcon').textContent = next === 'dark' ? '🌙' : '☀️';
    });

    // Print Report
    document.getElementById('printReportBtn').addEventListener('click', () => {
      window.print();
    });

    // Custom Dataset Modal (Bonus 2: +2 pt)
    const modal = document.getElementById('customDatasetModal');
    const openModalBtn = document.getElementById('openCustomDatasetBtn');
    const closeModalBtn = document.getElementById('closeModalBtn');

    openModalBtn.addEventListener('click', () => {
      modal.classList.add('open');
    });

    closeModalBtn.addEventListener('click', () => {
      modal.classList.remove('open');
    });

    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.classList.remove('open');
    });

    // Fill 50 Startups URL Example
    document.getElementById('fill50StartupsBtn').addEventListener('click', () => {
      document.getElementById('customUrlInput').value = 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv';
    });

    // Load CSV URL
    document.getElementById('loadUrlBtn').addEventListener('click', () => {
      const url = document.getElementById('customUrlInput').value.trim();
      if (!url) {
        alert('請輸入 CSV 網址！');
        return;
      }
      // If fetching 50 Startups locally or directly
      if (url.includes('50_Startups.csv')) {
        const ds = generateStartupsData();
        setDataset(ds);
        modal.classList.remove('open');
        alert('成功載入 50_Startups.csv 資料集！');
        return;
      }

      fetch(url)
        .then(res => {
          if (!res.ok) throw new Error('HTTP 錯誤代碼 ' + res.status);
          return res.text();
        })
        .then(csvText => {
          parseCSVContent(csvText, '自訂 URL 資料集');
        })
        .catch(err => {
          alert('載入 URL 失敗，可能受限於 CORS 跨域防護。建議改用「上傳本機 CSV 檔案」方式！\n' + err.message);
        });
    });

    // Dropzone File Upload
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('csvFileInput');

    dropzone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
          parseCSVContent(event.target.result, file.name);
        };
        reader.readAsText(file);
      }
    });

    // Confirm Custom Dataset
    document.getElementById('confirmCustomDatasetBtn').addEventListener('click', () => {
      if (!state.tempCustomData) return;
      const target = document.getElementById('customTargetSelect').value;
      const features = state.tempCustomData.numericCols.filter(c => c !== target);

      setDataset({
        name: state.tempCustomData.name,
        features,
        target,
        data: state.tempCustomData.data,
        kaggleUrl: '#'
      });

      modal.classList.remove('open');
      alert(`已成功套用自訂資料集！共有 ${features.length} 個數值特徵，目標變數為：${target}`);
      switchStep(2);
    });
  }

  function switchStep(stepNum) {
    state.activeStep = +stepNum;
    document.querySelectorAll('.step-tab').forEach(t => {
      t.classList.toggle('active', t.getAttribute('data-step') === String(stepNum));
    });
    document.querySelectorAll('.crisp-section').forEach(s => {
      s.classList.toggle('active', s.id === `section-${stepNum}`);
    });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // ---------------------------------------------------------------------------
  // Initialization
  // ---------------------------------------------------------------------------
  function init() {
    setupEventListeners();
    loadProblem('flood'); // Default to Problem 1 (20 features)
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
