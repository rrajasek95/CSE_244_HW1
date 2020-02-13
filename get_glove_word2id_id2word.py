import pickle as pkl
import os

if __name__ == '__main__':
    current_dir = os.getcwd()
    word2vec_filename = "glove.6B.300d.word2vec.txt"
    word2vec_path = os.path.join(current_dir, 'word2vec', word2vec_filename)

    idx2word = dict()
    word2idx = dict()

    with open(word2vec_path, 'r') as word2vec:
        # Ignore header
        word2vec.readline()

        for line in word2vec:
            word = line.split(" ")[0]

            idx = len(word2idx)
            word2idx[word] = idx
            idx2word[idx] = word

    processing_folder = os.path.join(os.getcwd(), 'processing')
    os.makedirs(processing_folder, exist_ok=True)
    dictionary_path = os.path.join(processing_folder, 'dictionary_300d.pkl')
    with open(dictionary_path, 'wb') as dictionary:
        pkl.dump({'id2word': idx2word, 'word2id': word2idx}, dictionary)