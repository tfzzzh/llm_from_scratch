import regex as re
from llm.pretokenization import find_chunk_boundaries, find_chunk_boundaries_on_chunksize
from typing import List, Optional, Dict, Tuple, Iterable, Iterator, Set
from multiprocessing import Pool
import pickle

GPT2SPLIT = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
CHARCODE = "utf-8"


def remove_special_tokens(
    doc: str, special_tokens: Optional[List[str]] = None
) -> List[str]:
    if special_tokens is None:
        return [doc]

    sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
    split_by = "|".join([re.escape(token) for token in sorted_special_tokens])
    docs_cleaned = re.split(split_by, doc)
    return docs_cleaned


def pretokenize(text: str) -> List[str]:
    if len(text) == 0: return []
    words = re.findall(GPT2SPLIT, text)
    return words


def calc_pretoken_freqs(docs: List[str]) -> Dict[str, int]:
    freqs = {}
    for doc in docs:
        words = re.findall(GPT2SPLIT, doc)
        for word in words:
            # if word == 'oxygen': print('found')
            freqs[word] = freqs.get(word, 0) + 1

    return freqs


class TokenizerTrainer:
    def __init__(self):
        self.word_cnts: Dict[str, int] = {}
        self.num_vocab: int = 256
        self.idx2token: List[bytes] = []  # update when new token generated
        self.token2idx: Dict[bytes, int] = {}  # update when new token generated
        self.pair_cnts: Dict[int, Dict[int, int]] = (
            {}
        )  # [idx -> {idx->cnt}] when (a,b) used as token, it shall removed from here
        self.tokenized_word_cnts: Dict[Tuple[int, ...], int] = {}

        # from adpapter
        self.vocab: Dict[int, bytes] = {}
        self.merges: List[Tuple[bytes, bytes]] = []
        self.reset()

    def reset(self):
        self.word_cnts: Dict[str, int] = {}
        # init using byte
        self.num_vocab: int = 256
        self.idx2token: List[bytes] = []
        self.token2idx: Dict[bytes, int] = {}
        for i in range(self.num_vocab):
            token = bytes([i])
            self.idx2token.append(token)
            self.token2idx[token] = i


        self.pair_cnts: Dict[int, Dict[int, int]] = {}
        self.tokenized_word_cnts: Dict[Tuple[int, ...], int] = {}
        self.vocab: Dict[int, bytes] = {}
        self.merges: List[Tuple[bytes, bytes]] = []

    def _train_check(self):
        # check size of vocab
        assert len(self.idx2token) == self.num_vocab
        assert len(self.token2idx) == self.num_vocab
        for i in range(self.num_vocab):
            assert self.token2idx[self.idx2token[i]] == i

        # check pair of cnt
        pair_cnts_correct: Dict[int, Dict[int, int]] = {}
        for key, value in self.tokenized_word_cnts.items():
            m = len(key)
            for i in range(m - 1):
                if key[i] not in pair_cnts_correct:
                    pair_cnts_correct[key[i]] = {}

                if key[i + 1] not in pair_cnts_correct[key[i]]:
                    pair_cnts_correct[key[i]][key[i + 1]] = 0

                pair_cnts_correct[key[i]][key[i + 1]] += value

        assert len(pair_cnts_correct) == len(self.pair_cnts)
        for idx in self.pair_cnts:
            assert len(self.pair_cnts[idx]) == len(pair_cnts_correct[idx])
            for jdx in self.pair_cnts[idx]:
                assert self.pair_cnts[idx][jdx] == pair_cnts_correct[idx][jdx]

        # check tokenized_word_cnts
        assert len(self.tokenized_word_cnts) == len(self.word_cnts)

    def from_word_cnts(self, word_cnts: Dict[str, int]):
        self.reset()
        self.word_cnts = word_cnts

        # init tokenized_word_cnts from word_cnts
        # use fact that [0-255] token_id and token_value is the same
        for word, cnt in word_cnts.items():
            tokened_word = tuple(word.encode(CHARCODE))
            assert tokened_word not in self.tokenized_word_cnts
            self.tokenized_word_cnts[tokened_word] = cnt

            # update pair_cnts using tokened_word
            m = len(tokened_word)
            if m == 1:
                continue
            for i in range(0, m - 1):
                token_a = int(tokened_word[i])
                token_b = int(tokened_word[i + 1])
                self.add_to_pair_counts(token_a, token_b, cnt)

        assert len(self.pair_cnts) > 0, "no token has length greater than 2"

    @staticmethod
    def from_doc(
        doc_path: str, vocab_size: int, special_tokens: List[str], num_processes=4
    ) -> Tuple[
        Dict[int, bytes], List[Tuple[bytes, bytes]], List[str], "TokenizerTrainer"
    ]:
        with open(doc_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_processes, "<|endoftext|>".encode(CHARCODE)
            )

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            # for start, end in zip(boundaries[:-1], boundaries[1:]):
            #     f.seek(start)
            #     chunk = f.read(end - start).decode(CHARCODE, errors="ignore")
            #     # Run pre-tokenization on your chunk and store the counts for each pre-token
            #     chunk_cleaned = remove_special_tokens(chunk, ['<|endoftext|>'])
            #     freqs = calc_pretoken_freqs(chunk_cleaned)

        results_fut = []
        results_ = []
        with Pool(num_processes) as pool:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                res = pool.apply_async(
                    TokenizerTrainer._count_job,
                    args=(doc_path, start, end, special_tokens),
                )
                results_fut.append(res)

            for res in results_fut:
                results_.append(res.get())

        # combine results
        freqs: Dict[str, int] = {}
        for freq in results_:
            for key, value in freq.items():
                freqs[key] = freqs.get(key, 0) + value

        trainer = TokenizerTrainer()
        trainer.from_word_cnts(freqs)
        trainer._train(vocab_size, special_tokens)
        return trainer.vocab, trainer.merges, special_tokens, trainer
    

    @staticmethod
    def _count_job(doc_path, start, end, special_tokens) -> Dict[str, int]:
        with open(doc_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode(CHARCODE, errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            chunk_cleaned = remove_special_tokens(chunk, special_tokens)
            freqs = calc_pretoken_freqs(chunk_cleaned)

        return freqs

    def _train(
        self,
        target_vocab_size,
        special_tokens: Optional[List[str]] = None,
        verbose=False,
    ):
        if special_tokens is None:
            special_tokens = []

        num_special_token = len(special_tokens)
        assert (
            self.num_vocab + num_special_token <= target_vocab_size
        ), f"target_vocab_size = {target_vocab_size} must at least {self.num_vocab} + {num_special_token}"

        while self.num_vocab < target_vocab_size - num_special_token:
            self._train_step()

        for special_token in special_tokens:
            self.add_special_token(special_token)

        self._train_check()

        if verbose:
            print(self.idx2token[256:])

        # update self.vocab
        # for i in range(256):
        #     self.vocab[i] = bytes([i])# .decode('latin-1').encode(CHARCODE)

        #     # special case
        #     # '/xe2'
        #     if i == 226:
        #         self.vocab[i] = bytes([i]).decode('latin-1').encode(CHARCODE)

        # for i in range(256, self.num_vocab):
        #     self.vocab[i] = self.idx2token[i].encode(CHARCODE)
        #     # [TODO] the test snapshot is not right since \xe2\x80 is not a correct utf-8 token
        #     # if self.vocab[i] == b"\xc3\xa2\xc2\x80":
        #     #     self.vocab[i] = b"\xe2\x80"
        for i in range(self.num_vocab):
            self.vocab[i] = self.idx2token[i]

    def _train_step(self):
        # select (a, b) with max cnts (when tie use lexical greater)
        max_cnt, a_idx, b_idx = self._select_max_freq_token_pair()
        if not (max_cnt != -1 and a_idx != -1 and b_idx != -1):
            print(self.tokenized_word_cnts)
            raise Exception("no new token found")

        # remove (a, b) from pair_cnts
        # update merge for pass the test
        new_tuple: Tuple[bytes, bytes] = (
            self.idx2token[a_idx],
            self.idx2token[b_idx],
        )

        self.merges.append(new_tuple)

        # insert new token a+b
        new_token = self.idx2token[a_idx] + self.idx2token[b_idx]
        new_token_id = len(self.idx2token)
        self.idx2token.append(new_token)
        self.token2idx[new_token] = new_token_id
        self.num_vocab += 1

        # update tokenized_word_cnts (via change keys)
        key_to_del: List[Tuple[int, ...]] = []
        pair_to_insert: List[Tuple[Tuple[int, ...], int]] = []
        self._adjust_token_pairs(a_idx, b_idx, new_token_id, key_to_del, pair_to_insert)

        # when key remove, I shall also update pair_cnt
        for key in key_to_del:
            value = self.tokenized_word_cnts[key]
            m = len(key)
            for i in range(0, m - 1):
                self.add_to_pair_counts(key[i], key[i + 1], -value)
            del self.tokenized_word_cnts[key]

        # when key insert, udpate pair_cnt
        for key, value in pair_to_insert:
            assert key not in self.tokenized_word_cnts
            self.tokenized_word_cnts[key] = value

            m = len(key)
            for i in range(0, m - 1):
                self.add_to_pair_counts(key[i], key[i + 1], value)

    def add_to_pair_counts(self, token_a: int, token_b: int, cnt_delta: int):
        # when delta < 0, token_a, token_b must in the dictionary
        if cnt_delta >= 0:
            if token_a not in self.pair_cnts:
                self.pair_cnts[token_a] = {}
            if token_b not in self.pair_cnts[token_a]:
                self.pair_cnts[token_a][token_b] = 0

        self.pair_cnts[token_a][token_b] += cnt_delta
        assert self.pair_cnts[token_a][token_b] >= 0

        # when cnt decrease to 0, it must be remove from the dict
        if self.pair_cnts[token_a][token_b] == 0:
            del self.pair_cnts[token_a][token_b]

        if len(self.pair_cnts[token_a]) == 0:
            del self.pair_cnts[token_a]

    def _adjust_token_pairs(
        self, a_idx, b_idx, new_token_id, key_to_del, pair_to_insert
    ):
        # loop over tokenized_word_cnts when (a_idx, b_idx) is
        # a slice of some key, one shall remove it and merge them
        # what about "aaaa"?'
        assert (
            self.idx2token[new_token_id]
            == self.idx2token[a_idx] + self.idx2token[b_idx]
        )
        # print(f"treat {a_idx}({self.idx2token[a_idx]}), {b_idx}({self.idx2token[b_idx]}) as {new_token_id}")

        for key, value in self.tokenized_word_cnts.items():
            m = len(key)
            # print(key)
            if m == 1:
                continue

            contain_new_token: bool = False
            for i in range(0, m - 1):
                if (key[i], key[i + 1]) == (a_idx, b_idx):
                    contain_new_token = True
                    break

            # when current key not contains a merge, noting to change
            if not contain_new_token:
                continue

            # when (key[i], key[i+1]) == (a_idx, b_idx)
            # change them to new_token_id
            # when a_idx, b_idx merged -> ?
            i = 0
            new_key = []
            while i < m:
                if i + 1 < m and (key[i], key[i + 1]) == (a_idx, b_idx):
                    new_key.append(new_token_id)
                    i += 2
                else:
                    new_key.append(key[i])
                    i += 1

            # udpate key_to_del
            key_to_del.append(key)

            # update pair_to_insert
            new_key = tuple(new_key)
            pair_to_insert.append((new_key, value))

        assert len(key_to_del) > 0, f"key_to_del={key_to_del}"

    def _select_max_freq_token_pair(self):
        max_cnt: int = -1
        a_idx: int = -1
        b_idx: int = -1
        for idx, freqs in self.pair_cnts.items():
            for jdx, freq in freqs.items():
                if freq > max_cnt:
                    max_cnt = freq
                    a_idx = idx
                    b_idx = jdx

                elif max_cnt == freq:
                    if (self.idx2token[idx], self.idx2token[jdx]) > (
                        self.idx2token[a_idx],
                        self.idx2token[b_idx],
                    ):
                        a_idx, b_idx = idx, jdx

        return max_cnt, a_idx, b_idx

    def add_special_token(self, special: str):
        if special in self.token2idx:
            raise Exception(f"speical token {special} shall not found in vocab")

        self.token2idx[special.encode(CHARCODE)] = self.num_vocab
        self.idx2token.append(special.encode(CHARCODE))
        self.num_vocab += 1


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None
    ):
        """Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.

        Args:
            vocab (Dict[int, bytes]): _description_
            merges (List[Tuple[bytes, bytes]]): _description_
            special_tokens (Optional[List[str]], optional): _description_. Defaults to None.
        """
        self.vocab: Dict[int, bytes] = vocab
        self.merges: List[Tuple[bytes, bytes]] = merges
        self.special_tokens: List[str] = (
            special_tokens if special_tokens is not None else []
        )

        # build inverse_vocab: when given bytes query its token_id
        self.inverse_vocab: Dict[bytes, int] = {}
        for token_id, token in vocab.items():
            self.inverse_vocab[token] = token_id

        # build merge_query: when given a pair of token_id query if it is a merge
        self.mergeid_query: Dict[Tuple[int, int], int] = {}
        self.merge_priority: Dict[Tuple[int, int], int] = {}
        for rank, (token_a, token_b) in enumerate(merges):
            token_a_id = self.inverse_vocab[token_a]
            token_b_id = self.inverse_vocab[token_b]
            token_merge_id = self.inverse_vocab[token_a + token_b]

            assert (token_a_id, token_b_id) not in self.mergeid_query

            self.mergeid_query[(token_a_id, token_b_id)] = token_merge_id
            self.merge_priority[(token_a_id, token_b_id)] = rank

        # build a inverse_vocab for special tokens
        self.inverse_vocab_specials: Dict[bytes, int] = {}
        for special_token in self.special_tokens:
            special_token = special_token.encode(CHARCODE)
            self.inverse_vocab_specials[special_token] = self.inverse_vocab[
                special_token
            ]

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        """constructs and return a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output)
        and (optionally) a list of special tokens.

        Args:
            vocab_filepath (str): _description_
            merges_filepath (str): _description_
            special_tokens (Optional[str], optional): _description_. Defaults to None.

        Returns:
            Tokenizer: _description_
        """
        import json
        with open(vocab_filepath, 'r') as file:
            vocab_str = json.load(file)
            vocab = {vocab_index: vocab_item.encode(CHARCODE, errors='replace') 
                     for vocab_item, vocab_index in vocab_str.items()}
            

        merges = []
        with open(merges_filepath, 'r') as file:
            for line in file:
                tokens = line.rstrip().split(" ")
                assert len(tokens) == 2
                tokens = (
                    tokens[0].encode(CHARCODE, errors="replace"),
                    tokens[1].encode(CHARCODE, errors="replace")
                )
                merges.append(tokens)

        # print(merges[:10])
        tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

        return tokenizer

    def encode(self, text: str) -> List[int]:
        """Encode an input text into a sequence of token IDs."""
        if len(self.special_tokens) == 0:
            words = pretokenize(text)

            tokens = []
            for word in words:
                tokens.extend(self._encode(word))

            return tokens
        
        # [!] handle overlapped special tokens like ['<end>', '<end><end>'], here we use greedy match
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = '|'.join([re.escape(stoken) for stoken in sorted_special_tokens])
        tokens = []
        latest_end = 0

        # split using special tokens
        for match_pos in re.finditer(special_pattern, text):
            start, end = match_pos.span()

            # handle text in [latest_end to start)
            words = pretokenize(text[latest_end : start])
            for word in words:
                tokens.extend(self._encode(word))

            # handle [start, end) (special token)
            stoken_id_lst = self._encode(text[start : end])
            assert (len(stoken_id_lst) == 1)
            tokens.append(stoken_id_lst[0])

            # update latest_end
            latest_end = end
        
        # handle remaining
        words = pretokenize(text[latest_end:])
        for word in words:
            tokens.extend(self._encode(word))

        return tokens


    def _encode(self, word: str) -> List[int]:
        """given a pretokenzied word return its tokenid"""
        # get byte repr of the word
        word_byte = word.encode(CHARCODE, errors="replace")

        # handle the case when word is special
        if word_byte in self.inverse_vocab_specials:
            return [self.inverse_vocab_specials[word_byte]]

        # iteratively merge
        tokens = list(self.inverse_vocab[bytes([elem])] for elem in word_byte)
        while len(tokens) > 1:
            i = 0
            best_idx = -1
            best_priority = float('inf')
            best_pair: Tuple[int, int] = (-1, -1)

            # find a position to merge
            while i + 1 < len(tokens):
                # case merge two keys
                if (
                    i + 1 < len(tokens)
                    and (tokens[i], tokens[i + 1]) in self.mergeid_query
                ):
                    token_pair = (tokens[i], tokens[i + 1])
                    priority = self.merge_priority[token_pair]
                    
                    if priority < best_priority:
                        best_priority = priority
                        best_pair = token_pair
                        best_idx = i

                i += 1

            # converge
            if best_idx == -1 : break

            tokens = tokens[:best_idx] + [self.mergeid_query[best_pair]] + tokens[best_idx+2:]

        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs

        Args:
            iterable (Iterable[str]): _description_

        Yields:
            Iterator[int]: _description_
        """
        for line in iterable:
            tokens = self.encode(line)
            for token in tokens:
                yield token

    def decode(self, ids: List[int]) -> str:
        """Decode a sequence of token IDs into text

        Args:
            ids (List[int]): _description_

        Returns:
            str: _description_
        """
        if len(ids) == 0:
            return ""
        
        byte_sequence = b""
        for token_id in ids:
            byte_sequence += self.vocab[token_id]

        return byte_sequence.decode(CHARCODE, errors='replace')

    def save_to_pickles(self, filepath: str):
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'special_tokens': self.special_tokens
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def from_pickle(cls, filepath: str) -> "Tokenizer":
        """Load tokenizer from pickle file"""
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        return cls(
            vocab=tokenizer_data['vocab'],
            merges=tokenizer_data['merges'],
            special_tokens=tokenizer_data['special_tokens']
        )