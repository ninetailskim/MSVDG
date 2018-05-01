#!/bin/bash
for i in $(seq 1 35) 
do 
python2 add_psf.py --ilmdb /data/Ytblmdb2/train${i}.lmdb --olmdb /data/Ytblmdb4/train_psf${i}.lmdb
done