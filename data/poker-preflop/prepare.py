import json
import random


def format_message(instruction, output):
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
    }


def prepare_train_data():
    with open('preflop_60k_train_set_prompt_and_label.json', 'r') as f:
        train_source = json.load(f)

    random.shuffle(train_source)
    split_idx = int(len(train_source) * 0.9)
    train_data = train_source[:split_idx]
    valid_data = train_source[split_idx:]

    # 寫入訓練資料
    with open('train.jsonl', 'w') as f:
        for item in train_data:
            json_line = format_message(item["instruction"], item["output"])
            f.write(json.dumps(json_line) + '\n')

    # 寫入驗證資料
    with open('valid.jsonl', 'w') as f:
        for item in valid_data:
            json_line = format_message(item["instruction"], item["output"])
            f.write(json.dumps(json_line) + '\n')

    print(f"Created train.jsonl with {len(train_data)} examples")
    print(f"Created valid.jsonl with {len(valid_data)} examples")


def prepare_test_data():
    with open('preflop_1k_test_set_prompt_and_label.json', 'r') as f:
        test_source = json.load(f)

    with open('test.jsonl', 'w') as f:
        for item in test_source:
            json_line = format_message(item["instruction"], item["output"])
            f.write(json.dumps(json_line) + '\n')

    print(f"Created test.jsonl with {len(test_source)} examples")


if __name__ == "__main__":
    prepare_train_data()
    prepare_test_data()
