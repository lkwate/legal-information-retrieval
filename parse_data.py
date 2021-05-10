import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from tqdm import tqdm
import argparse


nlp = en_core_web_sm.load()
tokenizer = nlp.tokenizer


def sentencizer(paragraph):
    sentences = list(nlp(paragraph).sents)
    return filter(lambda x : len(x) > 12, sentences)

def stream_generator_data(args, dataframe, text_column):
    data = dataframe[[text_column]]
    current_text, current_length = "", 0

    for row in tqdm(list(data.iterrows())):
        sentences = sentencizer(row[1][text_column])
        for sent in sentences:
            sent = str(sent)
            tokens = tokenizer(sent)
            current_length += len(tokens)
            current_sent += sent

            if current_length >= args.MAX_LEN_TOKEN:
                result, current_sent = current_sent, ""
                current_length = 0
                yield result

def transform(args):
    input_data = pd.read_csv(args.input_data_file)
    output_data = []

    for text in stream_generator_data(args, input_data, args.text_columns):
        output_data.append(text)

    output_data = pd.DataFrame(data=output_data, columns=['text'])
    output_data.to_csv(args.output_data_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameter description')
    parser.add_argument('--input_data_file', dest='input_data_file', type=str, required=True)
    parser.add_argument('--text_column', dest='text_column', type=str, required=True)
    parser.add_argument('--output_data_file', dest='output_data_file', type=str, required=True)
    parser.add_argument('--max_len_token', dest='max_len_token', str=int)

    args = parser.parse_args()
    
    transform(args)
