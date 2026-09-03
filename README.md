# magnet-loss-gate — 磁芯損耗 ML 預測＋物理驗收閘門（一日 demo）

用 [Princeton MagNet](https://www.princeton.edu/~minjie/magnet.html)（IEEE PELS MagNet Challenge 2023）的實測資料，
從 B(t) 波形特徵預測磁芯體積損耗，並讓**每個 ML 預測都附帶磁性元件工程師看得懂、否決得了的物理檢核**。

## 這個 demo 主張什麼

物理域 AI 的難點不在模型在驗收。LightGBM 打贏 Steinmetz 經驗公式不意外；
有價值的是驗收閘門（G1–G4）——它讓域專家能用自己的語言（正定性、頻率單調性、
對 Steinmetz 的偏離倍率）決定要不要信這個模型，而不是被要求信一個黑盒。

| 閘門 | 檢核 | 物理依據 |
|---|---|---|
| G1 | 預測損耗恆正 | 損耗不可能為負（log-target 使其結構性成立） |
| G2 | 頻率上升損耗不得下降 | 渦流與磁滯損耗隨 f 單調 |
| G3 | 峰值磁通上升損耗不得下降 | Steinmetz β>0 |
| G4 | 與 Steinmetz 預測偏離 >3× 的樣本逐筆列出 | 經驗公式是域內共識的 sanity bound |

方法論與 [polymer-tg-calibration](https://github.com/yschang1688) 同構：零真值情境下用物理規則驗收 ML 預測。

## 誠實範圍（一日 timebox）

- 單一磁材、未調參、75/25 隨機切分（無跨磁材泛化宣稱）
- 特徵只取 8 個物理可解釋量（f、T、B 峰值／波形形狀），刻意不用序列模型
- Steinmetz baseline 以全波形資料擬合（未按正弦子集分開），係數僅供 sanity bound
- MagNet Challenge 的正式指標是 95 分位相對誤差，本 demo 沿用（p50/p95/p99）

## 跑法

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# data/: 從 Princeton 頁面下載 final-training.zip 解壓
.venv/bin/python src/train.py --material N87 --data data/extracted
.venv/bin/mlflow ui   # 實驗追蹤：Steinmetz vs LightGBM、G1–G4 全數入 MLflow
```

資料授權與出處：Princeton MagNet open database（研究用途公開）。
