#!/bin/bash

# cudaのロード（環境によっては不要）
#module load cuda/12.4

### 各自の環境に合わせて変更してください #
#ALPHAFOLD3DIR="/home/hikari/alphafold3"    # AlphaFold3のコードのディレクトリ
HMMER3_BINDIR="/home/hikari/hmmer/bin" # HMMER3のバイナリディレクトリ
DB_DIR="/data1/hikari/public_databases"   # 配列・構造データベースのディレクトリ
MODEL_DIR="/home/hikari/models"         # モデルパラメータのディレクトリ
##########################################
### activate alphafold3's virtual environment
cd ~/alphafold3
#python3.11 ${ALPHAFOLD3DIR}/run_alphafold.py \
uv run run_alphafold.py \
    --jackhmmer_binary_path="${HMMER3_BINDIR}/jackhmmer" \
    --nhmmer_binary_path="${HMMER3_BINDIR}/nhmmer" \
    --hmmalign_binary_path="${HMMER3_BINDIR}/hmmalign" \
    --hmmsearch_binary_path="${HMMER3_BINDIR}/hmmsearch" \
    --hmmbuild_binary_path="${HMMER3_BINDIR}/hmmbuild" \
    --db_dir="${DB_DIR}" \
    --model_dir=${MODEL_DIR} \
    --json_path="AlphaFold_with_4G8A_single.json" \
    --output_dir="output"
