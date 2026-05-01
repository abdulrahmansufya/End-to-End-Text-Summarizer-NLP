import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from datasets import load_from_disk
from textSummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt
        ).to(device)

        seq2seq_data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model
        )

        # Load transformed/tokenized dataset
        dataset_samsum_pt = load_from_disk(self.config.data_path)

        trainer_args = Seq2SeqTrainingArguments(
            output_dir=self.config.root_dir,

            num_train_epochs=self.config.num_train_epochs,

            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,

            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,

            warmup_steps=self.config.warmup_steps,

            logging_steps=self.config.logging_steps,

            evaluation_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,

            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,

            predict_with_generate=self.config.predict_with_generate,
            generation_max_length=self.config.generation_max_length,
            generation_num_beams=self.config.generation_num_beams,

            fp16=torch.cuda.is_available(),
            report_to="none"
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=trainer_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"]
        )

        trainer.train()

        # Save model and tokenizer 
        save_path = os.path.join(self.config.root_dir, "distilbart-samsum-model")

        trainer.save_model(save_path)
        tokenizer.save_pretrained(save_path)