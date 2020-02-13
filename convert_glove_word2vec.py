import argparse

from gensim.test.utils import datapath, get_tmpfile

from gensim.scripts.glove2word2vec import glove2word2vec
import os
import shutil

if __name__ == '__main__':
    current_dir = os.getcwd()
    glove_file_path = os.path.join(current_dir, 'glove', 'glove.6B.300d.txt')
    glove_file = datapath(glove_file_path)

    word2vec_filename = "glove.6B.300d.word2vec.txt"
    word2vec_glove_file = get_tmpfile(word2vec_filename)
    glove2word2vec(glove_file, word2vec_glove_file)
    word2vec_path = os.path.join(current_dir, 'word2vec') 
    os.makedirs(word2vec_path, exist_ok=True)
    word2vec_file = os.path.join(word2vec_path, word2vec_filename)
    shutil.move(word2vec_glove_file, word2vec_file)

    print("GloVe processed! Now run get_glove_word2id_id2word.py")