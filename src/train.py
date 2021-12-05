import argparse
from transformers import (
    BertForTokenClassification, BertJapaneseTokenizer, BertConfig,
    TrainingArguments, Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split

import torch
import noyaki
import json


class Train:
    def __init__(self, args):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(args.model_name)
        self.args = args

    def run(self):
        features = self._load_from_json(self.args.training_data_path)
        train_data, val_data = train_test_split(features, test_size=0.2, random_state=123)
        train_data, test_data = train_test_split(train_data, test_size=0.1, random_state=123)
        self.label2id, id2label = self._create_label_vocab(features)

        config = BertConfig.from_pretrained(self.args.model_name, label2id=self.label2id, id2label=id2label)
        model = BertForTokenClassification.from_pretrained(self.args.model_name, config=config)

        args = TrainingArguments(output_dir=self.args.checkpoint_dir,
                                 do_train=True,
                                 do_eval=True,
                                 do_predict=True,
                                 per_device_train_batch_size=self.args.batch_size,
                                 per_device_eval_batch_size=self.args.batch_size,
                                 learning_rate=self.args.lr,
                                 num_train_epochs=self.args.num_epochs,
                                 evaluation_strategy="steps",
                                 eval_steps=self.args.save_freq,
                                 save_strategy="steps",
                                 save_steps=self.args.save_freq,
                                 load_best_model_at_end=True,
                                 )
        trainer = Trainer(model=model,
                          args=args,
                          data_collator=self._data_collator,
                          train_dataset=train_data,
                          eval_dataset=val_data,
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
                          )
        trainer.train()

        _, _, metrics = trainer.predict(test_data, metric_key_prefix="test")
        print(metrics)

        trainer.save_model(self.args.model_output_dir)

    def _load_from_json(self, path: str) -> list:
        json_dict = json.load(open(path, "r"))
        features = []
        for unit in json_dict:
            tokenized_text = self.tokenizer.tokenize(unit["text"])
            spans = []
            for entity in unit["entities"]:
                span_list = []
                span_list.extend(entity["span"])
                span_list.append(entity["type"])
                spans.append(span_list)
            label = noyaki.convert(tokenized_text, spans, subword="##")
            features.append({"x": tokenized_text, "y": label})
        return features

    def _data_collator(self, features: list) -> dict:
        x = [f["x"] for f in features]
        y = [f["y"] for f in features]
        inputs = self.tokenizer(x, return_tensors=None, padding='max_length', truncation=True, max_length=64, is_split_into_words=True)
        input_labels = []
        for labels in y:
            pad_list = [-100] * 64
            for i, label in enumerate(labels):
                pad_list.insert(i, self.label2id[label])
            input_labels.append(pad_list[:64])
        inputs['labels'] = input_labels
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in inputs.items()}
        return batch

    @staticmethod
    def _create_label_vocab(features: list) -> tuple:
        labels = [f["y"] for f in features]
        unique_labels = list(set(sum(labels, [])))
        label2id = {}
        for i, label in enumerate(unique_labels):
            label2id[label] = i
        id2label = {v: k for k, v in label2id.items()}
        return label2id, id2label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--model-name", type=str, default="cl-tohoku/bert-base-japanese-whole-word-masking")
    parser.add_argument("--model-output-dir", type=str, default="./dest")
    parser.add_argument("--training-data-path", type=str, default="./ner.json")
    parser.add_argument("--checkpoint-dir", type=str, default="./ckpt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-freq", type=int, default=100)
    parser.add_argument("--num-epochs", type=int, default=10)

    args, _ = parser.parse_known_args()

    train = Train(args)
    train.run()
