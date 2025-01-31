accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=32884 \
    scripts/train.py \
        --config './configs/quantized_images.yaml' \
        --exp_dir './experiments' \
        --data_dir './data'
