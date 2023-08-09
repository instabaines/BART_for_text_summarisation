# BART_for_text_summarisation

This repsository details codes and processes for processing data and trainng a model for summarization using LLM.

The model used in this case is a BART model.

The scripts can be run as standalone and can be run together. 

* processing and storing data
change the directories in process_and_store_data function in  preprocess/preprocess.py. Then do the following from the terminal
```bash
cd preprocess
python preprocess.py

```

* preprocessing and training the model
Modify the path_to_data in the main function in train/train.py. Point it to the direction of the raw data. 
Then run the following

```bash
cd train
python train.py
```