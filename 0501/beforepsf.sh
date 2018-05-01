#!/bin/bash
for i in $(seq 1 35) 
do 
python2 data_sampler.py --pattern '/data/Ytbdata/*_blurry.mp4' --lmdb /data/Ytblmdb2/train${i}.lmdb --num 5000
done