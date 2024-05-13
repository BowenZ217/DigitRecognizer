import os
import lzma
import pickle
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def unpickle_data(file):
    """
    Unpickle file using lzma and pickle.
    :param file: file path
    :return: data from file
    """
    if not os.path.exists(file):
        return None
    with lzma.open(file, 'rb') as fo:
        return pickle.load(fo)

def load_dataset(base_path, start_idx, end_idx, full=False, test_size=0.25):
    """
    Load and merge datasets from multiple pickle files.
    :param base_path: Base path pattern for the files
    :param start_idx: Start index of files
    :param end_idx: End index of files
    :param test_size: Fraction of data to be used as test set
    :return: Tuple containing train and test data and labels
    """
    X_list, Y_list = [], []
    for idx in range(start_idx, end_idx + 1):
        data_path = f"{base_path}_{idx}.pkl"
        data = unpickle_data(data_path)
        if data:
            X_list.append(data['data'])
            Y_list.extend(data['labels'])

    # Combine all data and labels into single arrays
    X = np.concatenate(X_list, axis=0)
    Y = np.array(Y_list)

    if full:
        return X, Y

    # Split into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X_train, Y_train, X_test, Y_test

class DigitDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels.astype(np.int64)  # Ensure labels are int64
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
