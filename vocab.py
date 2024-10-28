from collections import Counter
import torch

# Vocabulary Class
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return text.lower().split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized_text]

def build (file_path = "data/train/radiology/captions.txt", freq_threshold=5):
    # Build the vocabulary
    captions = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                _, caption = parts
                captions.append(caption)

    # Create and build vocabulary
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(captions)
    return vocab

def decode_caption(caption, vocab):
    """
    Convert a list of word indices into a readable caption string.
    
    Args:
        caption (list or tensor): List of word indices representing the caption.
        vocab (Vocabulary): Vocabulary object containing `itos` (index-to-string mapping).
    
    Returns:
        str: Decoded caption as a string.
    """
    # Ensure the caption is a list (convert from tensor if necessary)
    if isinstance(caption, torch.Tensor):
        caption = caption.tolist()

    # Map indices to words, excluding special tokens
    decoded_words = [vocab.itos.get(idx, "<unk>") for idx in caption if idx not in {vocab.stoi["<start>"], vocab.stoi["<end>"], vocab.stoi["<pad>"]}]
    
    # Join the words into a single string sentence
    decoded_sentence = ' '.join(decoded_words)
    
    return decoded_sentence
