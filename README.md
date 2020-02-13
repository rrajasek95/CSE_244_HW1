# CSE244 Assignment 1

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