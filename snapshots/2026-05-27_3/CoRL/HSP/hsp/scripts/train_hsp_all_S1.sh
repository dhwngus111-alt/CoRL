#!/bin/bash

# 얘네들이 train/train_overcooked_hsp.py 를 실행하는 거다.
# Policy Pool 후보들을 만드는 과정이다.

# Overcooked layout 하나 선택
# seed 1~36 반복
# MAPPO로 HSP Stage 1 정책 학습
# --use_hsp 켜고
# --w0를 랜덤 reward weight로 샘플링
# --w1는 거의 고정 baseline weight
# 학습 결과를 hsp/scripts/results 아래 저장


./hsp/s1/unident_s.sh  
./hsp/s1/random1.sh     
./hsp/s1/random3.sh     
./hsp/s1/distant_tomato.sh
./hsp/s1/many_orders.sh
