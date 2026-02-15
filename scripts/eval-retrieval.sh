export ENABLE_KERNELS=mamba2_ssm,sonicmoe,gru,rnn
export TRITON_PRINT_AUTOTUNING=1
export HF_HUB_OFFLINE=1

accelerate launch -m lm_eval \
    --model hf \
    --model_args dtype=bfloat16,pretrained=$1,max_length=32768 \
    --tasks fda,swde,based_nq_2048,based_triviaqa,squad_completion,based_drop \
    --metadata=metadata.json \
    --batch_size auto \
    --output_path $2
