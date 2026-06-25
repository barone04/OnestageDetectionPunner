#!/bin/bash
# Bootstrap GPU template (FPT) — chay tu repo ROOT:  bash scripts/setup.sh
# Env override:
#   DATA_URL=<gdrive_id|url>  bash scripts/setup.sh     # cai lib + tai data
set -e

DATA_URL="${DATA_URL:-}"      # FILE_ID gdrive (hoac URL). De trong -> bo qua tai data.

pip install -q -r requirements.txt

# Data KHONG nam trong git -> tai neu co DATA_URL va chua co san
if [ -n "$DATA_URL" ] && [ ! -d NewDeepfish ] && [ ! -d NewEtroplusMaculatus ]; then
  pip install -q gdown
  gdown --fuzzy "$DATA_URL" -O data.zip
  unzip -q data.zip -d . && rm data.zip
fi

python -c "import torch; print('CUDA:', torch.cuda.is_available())"
echo "OK. --data-path = folder TRUC TIEP chua images/ (data long thi la ./NewDeepfish/NewDeepfish)"
