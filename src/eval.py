import numpy as np
import argparse
import json
import noyaki
from seqeval.metrics import classification_report
from seqeval.scheme import BILOU
from transformers import BertJapaneseTokenizer, BertForTokenClassification


class Eval:
    def __init__(self, args):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(args.model_dir)
        self.model = BertForTokenClassification.from_pretrained(args.model_dir)
        self.args = args

    def run(self):
        features = self._load_from_json(self.args.training_data_path)
        y_true = []
        for unit in features:
            y_true.append(unit["y"][:64])
        y_pred = []
        for unit in features:
            y_pred.append(self._inference(unit["x"]))
        print(classification_report(y_true, y_pred, mode='strict', scheme=BILOU))

    def _inference(self, tokenized_text: list) -> list:
        inputs = self.tokenizer(tokenized_text, return_tensors="pt", padding='max_length', truncation=True, max_length=64, is_split_into_words=True)
        pred = self.model(**inputs).logits[0]
        pred = np.argmax(pred.detach().numpy(), axis=-1)
        labels = []
        for i, label in enumerate(pred):
            if i + 1 > len(tokenized_text):
                continue
            labels.append(self.model.config.id2label[label])
        return labels

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default="./dest")
    parser.add_argument("--training-data-path", type=str, default="./ner.json")

    args, _ = parser.parse_known_args()
    evaluate = Eval(args)
    evaluate.run()
