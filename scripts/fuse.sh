fine_tune_type="lora"
model="meta-llama/Llama-3.2-3B-Instruct"  # Add this line
job_name="${fine_tune_type}-Llama-3.2-3B-Instruct-lr-5"
adapter_path_base=../adapters/${job_name}
output_path_base=../models/${job_name}
hf_repo=weber50432/${job_name}
gguf_path=${job_name}.gguf

python -m mlx_lm.fuse \
    --model $model \
    --save-path $output_path_base \
    --adapter-path $adapter_path_base \
    --upload-repo $hf_repo \
    --export-gguf \
    --gguf-path $gguf_path 