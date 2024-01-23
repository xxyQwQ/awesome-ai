#!/bin/bash

if [ ! -d "./warehouse" ]; then
    echo "Please run under the root directory of this lab!"
    exit -1
fi

total_bombs=30
student=$(cat ./student-number.txt)
bomb_id=$(($student%$total_bombs+1))

cp ./warehouse/bomb-${bomb_id} ./bomb
