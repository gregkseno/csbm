accelerate launch \
    --num_processes=1 \
    --num_machines=1 \
    --mixed_precision='no' \
    --dynamo_backend='no' \
    --main_process_port=32489 \
    generate.py \
        --exp_path './experiments/quantized_images/uniform/dim_128_aplha_0.005_27.01.25_21:56:36' \
        --iterations 4 3 2
