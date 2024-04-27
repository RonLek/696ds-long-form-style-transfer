import json
import csv
from urllib.parse import urlparse

def extract_publication_name(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    publication_name = domain.split('.')[-2].capitalize()
    return publication_name

def process_json_data(json_data):
    data = []
    for obj in json_data:
        url = obj['url']
        content = obj['content']
        publication_name = extract_publication_name(url)
        data.append([url, publication_name, content])
    return data

def write_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['url', 'Publication_name', 'reference_doc'])
        writer.writerows(data)

def main():
    json_data = []
    with open('/Users/dishankj/Documents/696ds-long-form-style-transfer/pairwise_data/tech.pairs.text.jsonl', 'r') as file:
        for line in file:
            json_data.append(json.loads(line))

    processed_data = process_json_data(json_data)
    write_to_csv(processed_data, 'Prompting/data/reference_docs/tech_reference.csv')

if __name__ == '__main__':
    main()