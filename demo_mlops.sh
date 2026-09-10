#!/usr/bin/env bash
# 面試現場示範：什麼時候該重訓，讓資料決定，不靠人記。
#   bash demo_mlops.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

hr() { printf '\n\033[36m%s\033[0m\n' "── $1 ─────────────────────────────────"; }

hr "1／同一種磁材的新資料進來"
$PY src/drift.py --material "Material B" --batch same 2>/dev/null \
  | grep -E '"batch"|"worst_feature"|"psi_threshold"|"retrain_triggered"'

hr "2／換一種磁材的資料進來"
$PY src/drift.py --material "Material B" --batch other --other "Material E" 2>/dev/null \
  | grep -E '"batch"|"worst_feature"|"psi_threshold"|"retrain_triggered"'

hr "3／決策留在 MLflow，誰都能回頭查"
$PY - <<'EOF' 2>/dev/null
import mlflow, pandas as pd
mlflow.set_tracking_uri("sqlite:///mlflow.db")
df = mlflow.search_runs(experiment_names=["magnet-loss-gate"],
                        filter_string="tags.mlflow.runName LIKE '%drift%'",
                        order_by=["attributes.start_time DESC"], max_results=5)
cols = {"start_time": "時間", "params.batch": "進來的資料",
        "metrics.psi_max": "最大 PSI", "tags.action": "判斷"}
have = [c for c in cols if c in df.columns]
out = df[have].rename(columns=cols)
if "時間" in out: out["時間"] = pd.to_datetime(out["時間"]).dt.strftime("%m-%d %H:%M")
print(out.to_string(index=False))
EOF

hr "口白"
cat <<'EOF'
「重訓不是排程排出來的，是資料觸發的：同磁材 PSI 0.008 不動，
 換磁材 2.45 就標重訓，門檻 0.25 寫在管線裡。
 每一次判斷連同逐特徵的數字都留在 MLflow，
 三個月後有人問『為什麼那天重訓』，查得到。」
EOF
