accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=29004 \
    scripts/train.py \
        --config './configs/toy.yaml' \
        --exp_dir './experiments' \
        --data_dir './data'
