import wget
import os

from zipfile import ZipFile

if __name__ == '__main__':
    glove_folder = os.path.join(os.getcwd(), "glove")

    os.makedirs(glove_folder, exist_ok=True)
    wget.download("http://nlp.stanford.edu/data/glove.6B.zip")
    print("File downloaded! Now unzipping to", glove_folder)
    with ZipFile("glove.6B.zip", "r") as zipObj:
        zipObj.extractall("glove")

    print("GloVe downloaded! Now run convert_glove_word2vec.py")