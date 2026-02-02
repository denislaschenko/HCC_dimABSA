#!/bin/bash

source .venv/bin/activate

echo "--------------------------------------"
echo "STARTING RUN: LAPTOP DOMAIN"
echo "--------------------------------------"
export DOMAIN="laptop"
python src/subtask_3/train_subtask3_clean.py


if [ $? -ne 0 ]; then
    echo "CRITICAL ERROR: Laptop run failed! Stopping experiment."
    exit 1
fi

echo "--------------------------------------"
echo "STARTING RUN: RESTAURANT DOMAIN"
echo "--------------------------------------"
export DOMAIN="restaurant"
python src/subtask_3/train_subtask3_clean.py

if [ $? -ne 0 ]; then
    echo "CRITICAL ERROR: Restaurant run failed!"
    exit 1
fi

echo "--------------------------------------"
echo "All experiments finished successfully."
echo "You can now check the logs in outputs/subtask_1/logs/"
echo "--------------------------------------"