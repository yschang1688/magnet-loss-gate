# magnet-loss-gate — AI 驅動的磁性元件設計參數優化（附 MLOps 漂移觸發）

> 目的：對照一則電源 AI 工程師 JD 的前兩條職責，用公開實測資料做一輪端到端的可驗證實作。
> 資料：[Princeton MagNet](https://www.princeton.edu/~minjie/magnet.html)（IEEE PELS MagNet Challenge 2023，磁材匿名版 A–E）。
> 範圍：兩個晚上、單磁材、未調參——**方法論的遷移證明，不是產品**。

## JD 條目 → 本 repo 的對應交付

| JD 條目 | 本 repo 交付 | 進入點 |
|---|---|---|
| **1. AI 驅動的設計優化**：多物理域特徵提取＋ML 建模，進行電源設計參數優化（磁性元件…） | 從 B(t) 波形提取 8 個物理特徵 → 代理模型（LightGBM，單調約束）→ **在伏秒約束下掃描 f × 波形，找最低磁芯損耗的工作點**；每個候選過物理閘門 G1–G5 與分布內檢查 | `src/features.py`・`src/train.py`・`src/optimize.py` |
| **2. MLOps 平台自動化**：減少每次訓練模型時人工介入 | 訓練／優化／漂移三段全記 MLflow；**PSI 漂移檢查 > 0.25 自動標 retrain**，同磁材重抽樣不觸發、換磁材觸發——決策可稽核不靠人記 | `src/drift.py`・`mlflow ui` |
| 3. 智能控制策略（嵌入式 AI for MCU/DSP） | **未涵蓋**（誠實缺口） | — |

## 結果（Material B，n=7,400）

**代理模型 vs 經驗式**（p95 相對誤差，75/25 隨機切分）

| 模型 | p95 誤差 | G2/G3 單調性違規率 |
|---|---|---|
| Steinmetz 經驗式（全波形擬合，α=1.84／β=2.07） | 82.7% | — |
| LightGBM | 16.7% | 0.25% |
| **LightGBM＋單調約束（採用）** | 20.2% | **0%** |

單調約束多付 3.5 個百分點的誤差，換到「模型的錯誤都在物理允許範圍內」——這是設計優化能用的前提。

**設計參數優化**（`B_pk · f = K` 伏秒約束，掃 60 個頻率 × 3 種資料集內波形樣板 = 180 個候選）

| 情境 | 經驗式建議 | 閘門後最佳點 | 損耗節省 | 閘門攔下 |
|---|---|---|---|---|
| 低伏秒 K=5e3, 25 °C | 501 kHz（推到頻率上限） | **~250 kHz** | **26.5%** | 2/180 |
| 高伏秒 K=4e4, 90 °C | 501 kHz | **~446 kHz** | **18.4%** | **146/180**；裸模型最佳點落在 50 kHz、B_pk=0.80 T（鐵氧體飽和的兩倍）、離訓練分布 13.4（門檻 0.58）——**優化器鑽進了代理模型外推最錯的角落，被 G5 飽和與分布內檢查攔下** |

Steinmetz 因 α<β 永遠把設計推到頻率上限（邊界解）；代理模型找到內部最佳點。波形間（sine vs triangle）的差異在模型雜訊帶內，本 demo 不對波形下結論。

**漂移觸發**（`drift.py`）：同磁材重抽樣 PSI 最大 0.008 → 不重訓；換成 Material E 的波形 PSI 最大 2.45（b_pk）→ `retrain_triggered=true`。

## 閘門 G1–G5

| 閘門 | 檢核 | 依據 |
|---|---|---|
| G1 | 預測損耗恆正 | log-target 結構性成立 |
| G2 | 損耗隨 f 不下降 | 渦流／磁滯損耗單調（單調約束後結構性成立） |
| G3 | 損耗隨 B_pk 不下降 | Steinmetz β>0（同上） |
| G4 | 與 Steinmetz 偏離 >3× 逐筆列出 | 經驗式是域內共識的 sanity bound |
| G5 | B_pk ≤ B_sat（預設 0.40 T） | 飽和是設計規則不是資料規則——資料裡沒有、模型不會知道 |
| OOD | 候選到訓練集的 kNN 距離 ≤ 訓練自距離 95 分位 | 代理模型只在看過的區域可信 |

## 已知限制（會被追問的）

- 樹模型的代理曲面是分段常數（圖上的鋸齒）；優化採透明網格掃描，BO／平滑代理是下一步。
- 單磁材內隨機切分，無跨磁材泛化宣稱；Steinmetz 以全波形擬合，僅當 sanity bound。
- 伏秒約束把設計變數簡化為 (f, 波形)；實務還有繞組損、體積、熱——同框架可加目標與約束。

## 跑法

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# data/extracted/final-training/：從 Princeton 頁面下載 final-training.zip 解壓
.venv/bin/python src/train.py    --material "Material B" --data data/extracted/final-training
.venv/bin/python src/optimize.py --material "Material B" --k 4e4 --temp 90
.venv/bin/python src/drift.py    --material "Material B" --batch other --other "Material E"
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

方法論與 [polymer-tg-calibration](https://github.com/yschang1688) 同構：零真值情境下用物理規則驗收 ML 預測。
