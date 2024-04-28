import csv
import pandas as pd

def read_finance_reference(filename):
    reference_data = {}
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            url = row['url']
            publication_name = row['Publication_name']
            content = row['reference_doc']
            reference_data[url] = {'publication_name': publication_name, 'content': content}
    return reference_data

def process_finance_pairs(pairs_filename, reference_data, output_filename):
    pairs_df = pd.read_csv(pairs_filename)
    
    paired_data = []
    
    for _, row in pairs_df.iterrows():
        doc1_url = row.iloc[0]  # Access the first column by integer position
        doc2_url = row.iloc[1]  # Access the second column by integer position
        iou = row.iloc[2]  # Access the third column by integer position
        
        try:
            doc1_data = reference_data[doc1_url]
            doc2_data = reference_data[doc2_url]
            
            paired_data.append({
                'iou': iou,
                'paired_doc1': doc1_data['content'],
                'paired_doc2': doc2_data['content'],
                'publication_name_paired_doc1': doc1_data['publication_name'],
                'publication_name_paired_doc2': doc2_data['publication_name']
            })
        except KeyError:
            print(f"Skipping pair: {doc1_url}, {doc2_url}")
            continue
    
    output_df = pd.DataFrame(paired_data)
    output_df.to_csv(output_filename, index=False)

def main():
    reference_filename = 'Prompting/data/reference_docs/tech_reference.csv'
    pairs_filename = '/Users/dishankj/Documents/696ds-long-form-style-transfer/pairwise_data/tech.pairs.csv'
    output_filename = '/Users/dishankj/Documents/696ds-long-form-style-transfer/Prompting/data/paired_docs/tech_paired.csv'

    reference_data = read_finance_reference(reference_filename)
    process_finance_pairs(pairs_filename, reference_data, output_filename)

if __name__ == '__main__':
    main()