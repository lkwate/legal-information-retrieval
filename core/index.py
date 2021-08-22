import argparse
import json
import os
import faiss
import numpy as np
import torch
from tqdm import tqdm
from pyserini.dindex import DprDocumentEncoder, TctColBertDocumentEncoder, AnceDocumentEncoder, AutoDocumentEncoder
from encoder import LongAutoDocumentEncoder

def init_encoder(encoder:str, device:torch.device):
    encoder = encoder.lower()
    if "dpr" in encoder:
        return DprDocumentEncoder(encoder, device=device)
    elif 'tct_colbert' in encoder:
        return TctColBertDocumentEncoder(encoder, device=device)
    elif 'ance' in encoder:
        return AnceDocumentEncoder(encoder, device=device)
    elif 'sentence-transformers' in encoder:
        return AutoDocumentEncoder(encoder, device=device, pooling='mean', l2_norm=True)
    else:
        return LongAutoDocumentEncoder(encoder, device=device, l2_norm=True, pooling="mean")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, help='encoder name or path', required=True)
    parser.add_argument('--dimension', type=int, help='dimension of passage embeddings', required=False, default=768)
    parser.add_argument('--corpus', type=str,
                        help='directory that contains corpus files to be encoded, in jsonl format.', required=True)
    parser.add_argument('--index', type=str, help='directory to store brute force index of corpus', required=True)
    parser.add_argument('--batch', type=int, help='batch size', default=64)
    parser.add_argument('--shard-id', type=int, help='shard-id 0-based', default=0)
    parser.add_argument('--shard-num', type=int, help='number of shards', default=1)
    parser.add_argument('--device', type=str, help='device cpu or cuda [cuda:0, cuda:1...]', default='cuda:0')
    parser.add_argument('--title-delimiter', type=str, default=None)
    args = parser.parse_args()

    model = init_encoder(args.encoder, device=args.device)

    index = faiss.IndexFlatIP(args.dimension)

    if not os.path.exists(args.index):
        os.mkdir(args.index)

    ids = []
    texts = []
    titles = []
    for file in sorted(os.listdir(args.corpus)):
        file = os.path.join(args.corpus, file)
        if file.endswith('json') or file.endswith('jsonl'):
            print(f'Loading {file}')
            with open(file, 'r') as corpus:
                for idx, line in enumerate(tqdm(corpus.readlines())):
                    info = json.loads(line)
                    docid = info['id']
                    ids.append(docid)
                    text = info['contents'].strip()
                    if args.title_delimiter is None:
                        texts.append(text.lower())
                    else:
                        if args.title_delimiter == '\\n':
                            args.title_delimiter = '\n'
                        elif args.title_delimiter == '\\t':
                            args.title_delimiter = '\t'
                        title, text = text.lower().split(args.title_delimiter)
                        titles.append(title)
                        texts.append(text)

    total_len = len(texts)
    shard_size = int(total_len / args.shard_num)
    start_idx = args.shard_id * shard_size
    end_idx = min(start_idx + shard_size, total_len)
    if args.shard_id == args.shard_num - 1:
        end_idx = total_len

    with open(os.path.join(args.index, 'docid'), 'w') as id_file:
        for idx in tqdm(range(start_idx, end_idx)):
            id_file.write(f'{ids[idx]}\n')

    for idx in tqdm(range(start_idx, end_idx, args.batch)):
        text_batch = texts[idx: min(idx + args.batch, end_idx)]
        if len(titles) != 0:
            title_batch = titles[idx: min(idx + args.batch, end_idx)]
            embeddings = model.encode(text_batch, title_batch)
        else:
            embeddings = model.encode(text_batch)
        index.add(np.array(embeddings))
    faiss.write_index(index, os.path.join(args.index, 'index'))