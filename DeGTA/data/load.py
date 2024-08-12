import data
import os
import json


def load_data(dataset, use_dgl=False, use_text=False, use_gpt=False, seed=0):
    if dataset == 'cora':
        from data.data_utils.load_cora import get_raw_text_cora as get_raw_text
        num_classes = 7
    elif dataset == 'citeseer':
        from data.data_utils.load_citeseer import get_raw_text_citeseer as get_raw_text
        num_classes = 6
    elif dataset == 'pubmed':
        from data.data_utils.load_pubmed import get_raw_text_pubmed as get_raw_text
        num_classes = 3
    elif dataset == 'ogbn-arxiv':
        from data.data_utils.load_arxiv import get_raw_text_arxiv as get_raw_text
        num_classes = 40
    elif dataset == 'wikics':
        from data.data_utils.load_wikics import get_raw_text_wikics as get_raw_text
        num_classes = 10
    elif dataset == 'cornell':
        from data.data_utils.load_cornell import get_raw_text_cornell as get_raw_text
        num_classes = 5
    elif dataset == 'texas':
        from data.data_utils.load_texas import get_raw_text_texas as get_raw_text
        num_classes = 5
    elif dataset == 'wisconsin':
        from data.data_utils.load_wisconsin import get_raw_text_wisconsin as get_raw_text
        num_classes = 5
    else:
        exit(f'Error: Dataset {dataset} not supported')
    
    # use explanations as data augmentation (in the Future)
    if use_gpt:
        data, text = get_raw_text(use_text=False, seed=seed)
        folder_path = './datasets/gpt_responses/{}'.format(dataset)
        print(f"using gpt: {folder_path}")
        n = data.y.shape[0]
        text = []
        for i in range(n):
            filename = str(i) + '.json'
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                content = json_data['choices'][0]['message']['content']
                text.append(content)
    else:
        data, text = get_raw_text(use_text=True, seed=seed) 
      
    return data, text, num_classes