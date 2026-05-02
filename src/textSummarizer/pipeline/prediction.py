from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, pipeline
import torch


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

        self.device = 0 if torch.cuda.is_available() else -1

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        self.pipe = pipeline(
            "summarization",
            model=self.config.model_path,
            tokenizer=self.tokenizer,
            device= -1
        )


    def predict(self, text):
        gen_kwargs = {
            "max_length": 70,
            "min_length": 25,
            "num_beams": 6,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 4,
            "early_stopping": True
}

        output = self.pipe(text, **gen_kwargs)[0]["summary_text"]

        return output