from load_data import load_data_and_show
from transformers import AutoTokenizer
 from transformers import DataCollatorForSeq2Seq

'''
The preprocessing function needs to:

Prefix the input with a prompt so BART knows this is a summarization task. Some models capable of multiple NLP tasks require prompting for specific tasks.
Use a context manager with the as_target_tokenizer() function to parallelize tokenization of inputs and labels.
Truncate sequences to be no longer than the maximum length set by the max_length parameter
'''

def load_and_tokenize_data(path,model_name = 'facebook/bart-large'):
    
    data=load_data_and_show(path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def preprocess_function(examples):
        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples["body"]]
        model_inputs = tokenizer(inputs, max_length=256, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['abstract'], max_length=128, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_data= data.map(preprocess_function, batched=True) # batched is set to True to process multiple elements of the dataset at once

    return tokenized_data,tokenizer



def get_tokenized_data(path,model):
    tokenized_data,tokenizer = load_and_tokenize_data(path)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    return tokenized_data,data_collator
