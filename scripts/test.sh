job_name="total-llama-3.1-8B-Instruct"
model="meta-llama/Meta-Llama-3.1-8B-Instruct" 

python ../src/evaluate_gpu.py \
    --model $model \
    --max-length 200 \
    --test-file ../data/test/test.jsonl\
    --adapter-path ../adapters/${job_name}\
    --output-path ../evaluation/\
    --iter 0005000