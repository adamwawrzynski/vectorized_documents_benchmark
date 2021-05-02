import logging
import time
import pickle
import os
from models.benchmark_model import BenchmarkModel
from transformers import LongformerTokenizer, LongformerModel
from transformers.integrations import TensorBoardCallback
from transformers import LongformerForSequenceClassification, Trainer, TrainingArguments, AutoConfig, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
from utils.preprocess import clean_string_longformer
from datetime import date


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
        le = LabelEncoder()
        for text, label in zip(text_list, labels_list):
            self.dataset.append({"text": text, "label": label})

        labels = le.fit_transform([l["label"] for l in self.dataset])
        for index, entry in enumerate(self.dataset):
            entry["label"] = labels[index]

        self.input_length = input_length

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch):
        input_ = example_batch['text']
        label = example_batch['label']

        source = self.tokenizer.encode_plus(
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
        glob_att = torch.zeros(source["input_ids"].shape, device=src_mask.device)
        glob_att[:, 0] = 2

        return {"input_ids": source_ids.to(dtype=torch.long),
                "attention_mask": src_mask.to(dtype=torch.long),
                "labels": target.to(dtype=torch.long),
                "global_attention_mask": glob_att.to(dtype=torch.long),}


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
        dataset_name,
        num_categories=None,
        validation_split=0.2,
        verbose=1,
        epochs=10,
        batch_size=4,
        input_length=4096,
        save_directory="saved",
        model_name="longformer-doc",
    ):
        super().__init__()
        self.num_categories = num_categories
        self.dataset_name = dataset_name
        self.embedding_size = embedding_size
        self.validation_split = validation_split
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_length = input_length
        self.num_categories = num_categories
        self.save_directory = save_directory
        self.model_name = model_name

        if not os.path.exists(self.save_directory):
            os.mkdir(self.save_directory)

    def build_model(
        self
    ):
        super().build_model()
        self.tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
        self.model = LongformerForSequenceClassification.from_pretrained(
            "allenai/longformer-base-4096",
            num_labels=self.num_categories,
        )
        self.model.to("cuda")

    def train(
        self,
        x,
        y=None
    ):
        logging.info("Building vectorizer on " + self.__class__.__name__)
        t0 = time.time()

        processed_dataset = [clean_string_longformer(entry) for entry in x]

        train_dataset = load_custom_dataset(
            self.tokenizer,
            processed_dataset,
            y,
            "train",
            self.input_length,
        )
        test_dataset = load_custom_dataset(
            self.tokenizer,
            processed_dataset,
            y,
            "test",
            self.input_length,
        )

        today = date.today()
        date_string = today.strftime("%d_%m_%Y")
        time_string = today.strftime("%H_%M_%S")

        training_args = TrainingArguments(
            output_dir=f'./results/{date_string}',
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'./logs/{date_string}',
            load_best_model_at_end=True,
            fp16=False,
            fp16_opt_level="O2",
            evaluation_strategy="epoch",
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        trainer.train()
        self.model = trainer.model
        model_path = os.path.join(
            self.save_directory,
            self.dataset_name,
            f"{self.model_name}_{time_string}"
        )
        self.model.save_pretrained(model_path)
        self.model = LongformerModel.from_pretrained(model_path)
        self.model.to("cuda")
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
        processed_dataset = [clean_string_longformer(entry) for entry in dataset]

        cls_tokens = None
        for text in processed_dataset:
            input_ids = self.tokenizer.encode(
                text,
                max_length=self.input_length,
                padding='max_length',
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            input_ids = input_ids.cuda()

            # initialize to local attention
            attention_mask = torch.ones(
                input_ids.shape,
                dtype=torch.long,
                device=input_ids.device,
            )
            # initialize to global attention to be deactivated for all tokens
            global_attention_mask = torch.zeros(
                input_ids.shape,
                dtype=torch.long,
                device=input_ids.device,
            )
            # set global attention to <CLS> token
            global_attention_mask[:, 0] = 2 

            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )
            pooled_output = outputs.pooler_output.detach().cpu().numpy()

            if cls_tokens is not None:
                cls_tokens = np.concatenate((cls_tokens, pooled_output))
            else:
                cls_tokens = pooled_output


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
