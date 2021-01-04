import logging
import time
import pickle
import os
from models.benchmark_model import BenchmarkModel
from transformers import AutoTokenizer
from transformers import LongformerForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import numpy

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, text_list, labels_list, input_length):
        self.tokenizer = tokenizer
        self.dataset = list()
        self.dataset = list()
        for text, label in zip(text_list, labels_list):
            self.dataset.append({"text": text, "label": label})
        self.input_length = input_length

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch):
        input_ = example_batch['text']
        label = example_batch['label']

        source = self.tokenizer(
            input_,
            max_length=self.input_length, 
            padding='max_length',
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        target = torch.tensor(label)

        return source, target

    def __getitem__(self, index):
        source, target = self.convert_to_features(self.dataset[index])
        
        source_ids = source["input_ids"].squeeze(0)
        src_mask  = source["attention_mask"].squeeze(0)
        # token_type_ids = target["token_type_ids"].squeeze(0)

        return {"input_ids": source_ids.to(dtype=torch.long),
                "attention_mask": src_mask.to(dtype=torch.long),
                # "token_type_ids": token_type_ids.to(dtype=torch.long),
                "labels": target.to(dtype=torch.long)}


def load_custom_dataset(tokenizer, x, y, mode, input_length):
    dataset = dict()
    dataset["text"] = list()
    dataset["label"] = list()    
    for text, label in zip(x, y):
        dataset["text"].append(text)
        dataset["label"].append(label)

    train = list(range(0, int(len(dataset["text"])*0.8)))
    test = list(range(int(len(dataset["text"])*0.8), int(len(dataset["text"])*0.9)))
    val = list(range(int(len(dataset["text"])*0.9), len(dataset["text"])))

    data_dict = dict()
    data_dict["test"] = test
    data_dict["train"] = train
    data_dict["validation"] = val

    data_x, data_y = list(), list()
    for i in data_dict[mode]:
        data_x.append(dataset["text"][i])
        data_y.append(dataset["label"][i])

    return CustomDataset(tokenizer, data_x, data_y, input_length)


class LongformerBERTModel(BenchmarkModel):
    def __init__(
        self,
        embedding_size,
        num_categories=None,
        validation_split=0.2, 
        verbose=1,
        epochs=10,
        batch_size=1,
        input_length=4096,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.validation_split = validation_split
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_length = input_length

    def build_model(
        self
    ):
        super().build_model()
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")
        self.model.to("cuda")

    def train(
        self,
        x,
        y=None
    ):
        logging.info("Building vectorizer on " + self.__class__.__name__)
        t0 = time.time()

        train_dataset = load_custom_dataset(self.tokenizer, x, y, "train", self.input_length)
        test_dataset = load_custom_dataset(self.tokenizer, x, y, "test", self.input_length)
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )
        trainer.train()
        self.model = trainer.model
        elapsed = (time.time() - t0)
        logging.info("Done in %.3fsec" % elapsed)

    def fit(
        self,
        x,
        y
    ):
        logging.info("Training kNN classifier")
        embedded_x = self.preprocess_data(x, y)
        return self.clf.fit(embedded_x ,y)

    def preprocess_data(
        self,
        dataset,
        y_dataset
    ):
        cls_tokens = None
        for text in dataset:
            input_ids = self.tokenizer.encode(
                input_,
                max_length=self.input_length, 
                padding='max_length',
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            outputs = self.model(input_ids)[1].detach().cpu().numpy()
            if cls_tokens is not None:
                cls_tokens = np.concatenate((cls_tokens, out_tokens))
            else:
                cls_tokens = out_tokens
        return cls_tokens

    def save(
        self,
        path
    ):
        logging.info("Saving " + self.__class__.__name__)
        combined_path = os.path.join(path, self.__class__.__name__)
        self.model.save_pretrained(combined_path)
        self.tokenizer.save_pretrained(combined_path)
        pickle.dump(self.clf,
            open(combined_path + "_clf.pickle", 'wb'))

    def load(
        self,
        path
    ):
        logging.info("Loading " + self.__class__.__name__)
        combined_path = os.path.join(path, self.__class__.__name__)
        self.model.from_pretrained(combined_path)
        self.tokenizer.from_pretrained(combined_path)
        self.clf = pickle.load(
            open(combined_path + "_clf.pickle", 'rb'))

    def can_load(
        self,
        path
    ):
        combined_path = os.path.join(path, self.__class__.__name__)
        return os.path.isfile(combined_path + "_clf.pickle")