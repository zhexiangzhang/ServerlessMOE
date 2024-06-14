#!/bin/bash

new_tokens=6

batch_size=1

while [ $batch_size -le 512 ]
do
    echo "Running: python3 hg.py $batch_size $new_tokens"
    python3 hg.py $batch_size $new_tokens
    batch_size=$(( batch_size * 2 )) 
done
