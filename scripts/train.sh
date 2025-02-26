# Install unbuffer if needed: brew install expect
fine_tune_type="lora"
model="meta-llama/Meta-Llama-3-8B-Instruct"  # Add this line
job_name="${fine_tune_type}-Meta-Llama-3-8B-Instruct-lr-6"
data_path=../data/poker-total
output_path_base=../adapters/${job_name}
adapter_file=../adapters/${job_name}/adapters.safetensors
log_dir="../logs"
mkdir -p "$log_dir"
log_file="$log_dir/${job_name}.log"

# Log start time
echo "Start time: $(date)" | tee "$log_file"

unbuffer \
python -m mlx_lm.lora \
    --train \
    --test \
    --fine-tune-type $fine_tune_type \
    --adapter-path $output_path_base \
    --resume-adapter-file $adapter_file\
    --model $model \
    --data $data_path \
    --iters 5000 \
    --num-layers 16 \
    --batch-size 2 \
    --val-batches 25 \
    --test-batches 100 \
    --steps-per-report 100 \
    --steps-per-eval 100 \
    --learning-rate 0.000001 \
    2>&1 | tee -a "$log_file"
    # --config "./config.yaml" \

echo "End time: $(date)" | tee -a "$log_file"