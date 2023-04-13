import json
import random

# Set the number of folds
num_folds = 5

# Read in the original jsonl file
data = []
with open('fake_news.jsonl') as f:
    for line in f:
        data.append(json.loads(line))

# Shuffle the data randomly
random.shuffle(data)

# Calculate the size of each fold
fold_size = len(data) // num_folds

# Split the data into folds
for fold in range(num_folds):
    # Calculate the start and end indices for the fold
    start = fold * fold_size
    end = (fold + 1) * fold_size if fold < num_folds - 1 else len(data)
    
    # Write the data for the fold to a new jsonl file
    with open(f'fake_news_fold_{fold}.jsonl', 'w') as f:
        for item in data[start:end]:
            f.write(json.dumps(item) + '\n')4