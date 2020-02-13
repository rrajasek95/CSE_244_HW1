# CSE244 Assignment 1

Code for Homework 1 of _CSE 244: Machine Learning for NLP_ taught by Dr. Dilek Hakkani-Tur at University of California, Santa Cruz.

## 
Run the following commands to setup GloVe (for LSTM training):
```
python download_glove.py
python convert_glove_word2vec.py
python get_glove_word2id_id2word.py
```

To run the train step, check `train_model.py` and set the appropriate main function to train the model and run `python train_model.py`
```
if __name__ == '__main__':
    train_multilabel_mlp()
```

For e.g. to train LSTM instead, change the function to:
```
if __name__ == '__main__':
    train_birnn()
```

To run the test data evaluation, run `python model_evaluation.py`. Change the main block to evaluate some other model instead. Note that the model must be trained and be present under `saved_models` folder in order to run the evaluation.

To run inference, run `inference.py`. Change the main block to evaluate some other model.

Model hyperparameters are stored in `config.py`, so please review them.

# Potential contributions

I strongly believe that one should fall in love with the problem, not the solution. I'm looking to collect more ideas and approaches towards handling the task plus some code cleanup and organization help. If you're looking to contribute towards improving this task, please open an issue (and potentially tie a PR to that issue). I am creating a bucket list of improvements I'd like to see for this repo.