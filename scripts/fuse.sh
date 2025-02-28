fine_tune_type="lora"
model="meta-llama/Meta-Llama-3.1-8B-Instruct"  # Add this line
job_name="${fine_tune_type}-Meta-Llama-3.1-8B-Instruct"
adapter_path_base=../adapters/${job_name}
output_path_base=../models/${job_name}
hf_repo=weber50432/${job_name}
gguf_path=${job_name}-ggml-model-f16.gguf

python -m mlx_lm.fuse \
    --model $model \
    --save-path $output_path_base \
    --adapter-path $adapter_path_base \
    --upload-repo $hf_repo \
    --export-gguf \
    --gguf-path $gguf_path 