fine_tune_type="lora"
model="meta-llama/Llama-3.2-3B-Instruct"  # choose the model to fuse
job_name="${fine_tune_type}-Llama-3.2-3B-Instruct" #fine-tuned model name
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
    --de-quantize\
    --gguf-path $gguf_path 