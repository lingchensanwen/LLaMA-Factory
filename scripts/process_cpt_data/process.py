import json
import random

def split_jsonl_by_text_field(
    input_file,
    train_file,
    val_file,
    test_file,
    train_ratio=0.9,
    val_ratio=0.05,
    test_ratio=0.05,
    seed=42
):

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    structured_data = [{"text": json.loads(line.strip())["text"]} for line in lines]
    
    random.seed(seed)
    random.shuffle(structured_data)
    
    total_count = len(structured_data)
    train_end = int(train_ratio * total_count)
    val_end = train_end + int(val_ratio * total_count)

    train_data = structured_data[:train_end]
    val_data = structured_data[train_end:val_end]
    test_data = structured_data[val_end:] 
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    with open(val_file, 'w', encoding='utf-8') as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    with open(test_file, 'w', encoding='utf-8') as f:
        for entry in test_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print("Splitting complete!")
    print(f"  Train set: {len(train_data)} rows")
    print(f"  Val set:   {len(val_data)} rows")
    print(f"  Test set:  {len(test_data)} rows")


if __name__ == "__main__":
    input_file_path = "/scratch/07144/yw23374/data/astro_code_v1.jsonl"
    train_file_path = "/scratch/07144/yw23374/data/structured_cpt_train.jsonl"
    val_file_path   = "/scratch/07144/yw23374/data/structured_cpt_val.jsonl"
    test_file_path  = "/scratch/07144/yw23374/data/structured_cpt_test.jsonl"
    
    split_jsonl_by_text_field(
        input_file=input_file_path,
        train_file=train_file_path,
        val_file=val_file_path,
        test_file=test_file_path,
        train_ratio=0.9,
        val_ratio=0.05,
        test_ratio=0.05,
        seed=42
    )
