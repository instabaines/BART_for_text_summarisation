from datasets import load_dataset
import os
import datasets
import random
import pandas as pd
from IPython.display import display, HTML

def load_data_and_show(path,split_data=True, show_data=True):

    _,extension = os.path.splitext(path)
    data = load_dataset(extension,data_files=path)
    # data=data['train']
    if split_data:
        data = data.train_test_split(test_size=0.2) # splitting into train and test

    if show_data:
        show_random_elements(data["train"])
    return data



def show_random_elements(dataset, num_examples=5):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)

    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    display(HTML(df.to_html()))