from llm.tokenizer import Tokenizer, TokenizerTrainer
import numpy as np

def train_tokenizer(vocab_size, write_path):
    print("============train_tokenizer============")
    data_path = 'data/TinyStoriesV2-GPT4-train.txt'
    vocab, merges,  special_tokens, _ = TokenizerTrainer.from_doc(data_path, vocab_size=vocab_size, special_tokens=['<|endoftext|>'], num_processes=4)
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    tokenizer.save_to_pickles(write_path)
    print(f"===========tokenizer saved to {write_path}=====")


if __name__ == '__main__':
    vocab_size = 10000
    tk_path = f'./tokenizer_{vocab_size}.pkl'
    train_tokenizer(vocab_size, tk_path)

    # load tokenizer
    print(f"load tokenizer from {tk_path}")
    tokenizer = Tokenizer.from_pickle(tk_path)

    # tokenize
    train_data_path = 'data/TinyStoriesV2-GPT4-train.txt'
    print(f'start tokenize train data: {train_data_path}')
    tokens = []
    with open(train_data_path, 'r') as file:
        for token in tokenizer.encode_iterable(file):
            tokens.append(token)

    tokens = np.array(tokens, dtype=np.int64)

    # store
    write_dataset_path = 'data/TinyStoriesV2-GPT4-train.npy'
    print(f"store tokens to {write_dataset_path}")
    with open(write_dataset_path, 'wb') as file:
        np.save(file, tokens)