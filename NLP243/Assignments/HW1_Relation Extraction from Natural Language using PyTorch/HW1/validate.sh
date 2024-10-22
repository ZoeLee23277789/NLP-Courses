#!/bin/bash

{
    unzip hw1.zip -d new_dir  # 解壓縮到新資料夾
    cd new_dir  # 進入新資料夾
    python -m venv venv  # 建立虛擬環境
    source venv/bin/activate  # 啟動虛擬環境
    pip install -r requirements.txt  # 安裝依賴

    echo '========== start running =========='  # 開始執行程式
    python run.py hw1_train.csv hw1_test.csv submission.csv

} 2>&1 | tee record.txt  # 記錄所有輸出
