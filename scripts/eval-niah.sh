export ENABLE_KERNELS=mamba2_ssm,sonicmoe,gru,rnn
export TRITON_PRINT_AUTOTUNING=1
export HF_HUB_OFFLINE=1

accelerate launch -m lm_eval \
    --model hf \
    --model_args dtype=bfloat16,pretrained=$1,max_length=32768 \
    --tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multiquery,niah_multivalue \
    --metadata=metadata.json \
    --batch_size auto \
    --output_path $2
