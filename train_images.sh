accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=32883 \
    scripts/train.py \
        --config './configs/images.yaml' \
        --exp_dir './experiments' \
        --data_dir './data'
