# Install unbuffer if needed: brew install expect
job_name="total-DeepSeek-R1-Distill-Llama-8B"
model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # Add this line
data_path=../data/poker-total
output_path_base=../adapters/${job_name}
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