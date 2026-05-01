import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        
        self.max_input_length = 512
        self.max_target_length = 96


    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(
            example_batch["dialogue"],
            max_length=self.max_input_length,
            truncation=True
        )

        target_encodings = self.tokenizer(
            text_target=example_batch["summary"],
            max_length=self.max_target_length,
            truncation=True
        )

        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"]
        }


    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)

        dataset_samsum_pt = dataset_samsum.map(
            self.convert_examples_to_features,
            batched=True
        )

        dataset_samsum_pt.save_to_disk(
            os.path.join(self.config.root_dir, "samsum_dataset")
        )