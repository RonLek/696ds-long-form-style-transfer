import csv
import pandas as pd
from urllib.parse import urlparse
import json

def get_content_for_url(jsonl_file, target_url):
    # Open the JSONL file
    with open(jsonl_file, 'r') as file:
        # Read each line (JSON object) in the file
        for line in file:
            # Parse the JSON object
            data = json.loads(line)
            # Check if the "url" key matches the target URL
            if data.get("url") == target_url:
                return data.get("content")  # Return the content if URL matches
    return None  # Return None if URL not found in the file

domains = ['tech', 'entertainment', 'games', 'food', 'finance']
dataset = []

for d in domains:
    df = pd.read_csv(f'v2/{d}.pairs.csv', header=None, names=['pair1', 'pair2', 'iou'])
    df.sort_values(by=['iou'], inplace=True, ascending=False)
    df.reset_index(drop=True, inplace=True)

    for index, row in df.iterrows():
        print(f'{d} {index}')
        content1 = get_content_for_url(f'v2/{d}.pairs.text.jsonl', row['pair1'])
        content2 = get_content_for_url(f'v2/{d}.pairs.text.jsonl', row['pair2'])
        pub1 = urlparse(row['pair1']).netloc.split('.')[-2]
        pub2 = urlparse(row['pair2']).netloc.split('.')[-2]
        dataset.append({'pair1': row['pair1'], 'pair2': row['pair2'], 'iou': row['iou'], 'pub1': pub1, 'pub2': pub2, 'content1': content1, 'content2': content2})
        if index == 19:
            break
    print(f'{d} done!')

res = pd.DataFrame(dataset)
res.to_csv('dataset.csv', index=False)

