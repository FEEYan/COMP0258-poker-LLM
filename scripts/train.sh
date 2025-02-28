# Install unbuffer if needed: brew install expect
fine_tune_type="lora"
model="meta-llama/Meta-Llama-3.1-8B-Instruct"  # Add this line
job_name="${fine_tune_type}-Meta-Llama-3.1-8B-Instruct-lr-5"
data_path=../data/poker-total
output_path_base=../adapters/${job_name}
# adapter_file=../adapters/lora-Meta-Llama-3-8B-Instruct-lr-5/adapters.safetensors
log_dir="../logs"
mkdir -p "$log_dir"
log_file="$log_dir/${job_name}.log"

# Log start time
echo "Start time: $(date)" | tee "$log_file"

unbuffer \
python -m mlx_lm.lora \
    --train \
    --test \
    --adapter-path $output_path_base \
    --fine-tune-type $fine_tune_type \
    --model $model \
    --data $data_path \
    --iters 5000 \
    --num-layers 16 \
    --batch-size 2 \
    --val-batches 25 \
    --test-batches 100 \
    --steps-per-report 100 \
    --steps-per-eval 100 \
    --learning-rate 0.00001 \
    2>&1 | tee -a "$log_file"
    # --resume-adapter-file $adapter_file\
    # --config "./config.yaml" \

echo "End time: $(date)" | tee -a "$log_file"