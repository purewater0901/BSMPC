#!/bin/bash

seeds=(
    0
    1
    2
)
envs=(
    config/dmcontrol/state/humanoid_walk/tdmpc.yaml
    config/dmcontrol/state/humanoid_stand/tdmpc.yaml
)

for env in ${envs[*]}
do
    for seed in ${seeds[*]}
    do
	    python main.py \
        --config $env \
	    --seed $seed
    done
done
