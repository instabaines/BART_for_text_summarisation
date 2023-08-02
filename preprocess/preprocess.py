from load_data import load_data_and_show
from transformers import AutoTokenizer

tokenizer = None

def load_and_tokenize_data(path,model_name = 'facebook/bart-large'):
    global tokenizer
    data=load_data_and_show(path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_data= data.map(preprocess_function, batched=True)

    return tokenized_data





def preprocess_function(examples):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["body"]]
    model_inputs = tokenizer(inputs, max_length=256, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['abstract'], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs