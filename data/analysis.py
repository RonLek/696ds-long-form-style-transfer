import json
from urllib.parse import urlparse
import matplotlib.pyplot as plt
import numpy as np

def get_publications_dict(domain):
    pub_dict = {}
    with open(f'v2/non_parallel/{domain}/data.000000000000.jsonl') as f:
        data = f.readlines()
        for line in data:
            jsonline = json.loads(line)
            domain_name = urlparse(jsonline['url']).netloc.split('.')[-2]
            if domain_name not in pub_dict.keys():
                pub_dict[domain_name] = {'count': 1, 'avg_tokens': len(jsonline['text'])}
            else:
                pub_dict[domain_name]['count'] += 1
                pub_dict[domain_name]['avg_tokens'] = (pub_dict[domain_name]['avg_tokens'] * (pub_dict[domain_name]['count'] - 1) + len(jsonline['text'])) / pub_dict[domain_name]['count']
    return pub_dict

def get_text_length_distribution(domain):
    text_list = []
    with open(f'v2/non_parallel/{domain}/data.000000000000.jsonl') as f:
        data = f.readlines()
        for line in data:
            text_list.append(len(json.loads(line)['text']))
   
    text_list = [l for l in text_list if l < 10000]
    # Define bins
    bin_edges = np.arange(0, max(text_list) + 100, 50)  # Adjust the bin width as needed

    # Compute histogram
    hist, _ = np.histogram(text_list, bins=bin_edges)

    # Plotting
    plt.bar(bin_edges[:-1], hist, width=100)  # Use width=10 to match bin width
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.xticks(np.arange(0, max(text_list) + 1, 1000))  # Adjust step size as needed
    plt.savefig(f'{domain}_length_distribution.png')


def bar_plot(data_dict, xlabel, ylabel, title, save_fig):
    plt.bar(data_dict.keys(), [val['count'] for val in data_dict.values()])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(save_fig)
    
    plt.close()

domains = ['entertainment', 'finance', 'food', 'games', 'tech']
domain_dict = {}
for d in domains:
    pub_dict = get_publications_dict(d)
    
    print(f'Publication dict for {d}:', pub_dict)
    bar_plot(pub_dict, 'Publications', 'Number of Articles', f'Distribution of Articles per Publication ({d})', f'{d}_distribution.png')
    
    domain_dict[d] = {'count': sum([val['count'] for val in pub_dict.values()])}

    get_text_length_distribution(d)

print('Domain dict: ', domain_dict)
bar_plot(domain_dict, 'Domain', 'Number of Articles', 'Distribution of Articles per Domain', 'domain_distribution.png')
