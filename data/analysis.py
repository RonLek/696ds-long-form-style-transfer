import json
from urllib.parse import urlparse
import matplotlib.pyplot as plt

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

print('Domain dict: ', domain_dict)
bar_plot(domain_dict, 'Domain', 'Number of Articles', 'Distribution of Articles per Domain', 'domain_distribution.png')
