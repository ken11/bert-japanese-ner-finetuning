import numpy as np
import argparse
from transformers import BertJapaneseTokenizer, BertForTokenClassification


class Inference:
    def __init__(self, args):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(args.model_dir)
        self.model = BertForTokenClassification.from_pretrained(args.model_dir)

    def run(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        inputs = self.tokenizer(tokenized_text, return_tensors="pt", padding='max_length', truncation=True, max_length=64, is_split_into_words=True)
        pred = self.model(**inputs).logits[0]
        pred = np.argmax(pred.detach().numpy(), axis=-1)
        labels = []
        for i, label in enumerate(pred):
            if i + 1 > len(tokenized_text):
                continue
            labels.append(self.model.config.id2label[label])
        print(tokenized_text)
        print(labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default="./dest")
    parser.add_argument("--input-text", type=str, default="田中さんの会社の社長は鈴木さんです")

    args, _ = parser.parse_known_args()
    evaluate = Inference(args)
    evaluate.run(args.input_text)
