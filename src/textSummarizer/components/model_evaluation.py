from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import torch
import pandas as pd
from tqdm import tqdm
import evaluate
from textSummarizer.entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """Split a list into smaller batch-sized chunks."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i: i + batch_size]


    def calculate_metric_on_test_ds(
        self,
        dataset,
        metric,
        model,
        tokenizer,
        batch_size=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        column_text="dialogue",
        column_summary="summary"
    ):
        dialogue_batches = list(
            self.generate_batch_sized_chunks(dataset[column_text], batch_size)
        )

        target_batches = list(
            self.generate_batch_sized_chunks(dataset[column_summary], batch_size)
        )

        model.eval()

        for dialogue_batch, target_batch in tqdm(
            zip(dialogue_batches, target_batches),
            total=len(dialogue_batches)
        ):
            inputs = tokenizer(
                dialogue_batch,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                summary_ids = model.generate(
                    **inputs,
                    length_penalty=0.8,
                    num_beams=4,
                    max_length=96
                )

            decoded_summaries = tokenizer.batch_decode(
                summary_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            decoded_summaries = [summary.strip() for summary in decoded_summaries]
            target_batch = [summary.strip() for summary in target_batch]

            metric.add_batch(
                predictions=decoded_summaries,
                references=target_batch
            )

        score = metric.compute(use_stemmer=True)
        return score


    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path
        ).to(device)

        dataset_samsum_pt = load_from_disk(self.config.data_path)

        rouge_metric = evaluate.load("rouge")

        score = self.calculate_metric_on_test_ds(
            dataset=dataset_samsum_pt["test"],
            metric=rouge_metric,
            model=model,
            tokenizer=tokenizer,
            batch_size=2,
            device=device,
            column_text="dialogue",
            column_summary="summary"
        )

        rouge_dict = {
            "rouge1": score["rouge1"],
            "rouge2": score["rouge2"],
            "rougeL": score["rougeL"],
            "rougeLsum": score["rougeLsum"]
        }

        df = pd.DataFrame(rouge_dict, index=["distilbart"])
        df.to_csv(self.config.metric_file_name)