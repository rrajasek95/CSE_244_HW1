from torch.utils.data import Dataset

class MovieRelationDataset(Dataset):
    def __init__(self, input_df, vectorizer):
        self.input_df = input_df
        self._vectorizer = vectorizer


    def __getitem__(self, index):
        row = self.input_df.iloc[index]

        x_input = self._vectorizer.vectorize_input(row.UTTERANCE)
        y_target = self._vectorizer.vectorize_output(row.RELATIONS)
        return {
            'x_data': x_input,
            'y_target': y_target
        }

    def __len__(self):
        return len(self.input_df)

    def get_num_batches(self, batch_size):
        return len(self) // batch_size