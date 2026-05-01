from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np
from typing import List

class Onnx_eval:

    def pad_sequence(self, input_ids_list: List[np.ndarray], padding_value: int = 0):
        max_len = max(len(sequence) for sequence in input_ids_list)
        padded_sequences = np.full((len(input_ids_list), max_len), padding_value)
        for i, sequence in enumerate(input_ids_list):
            padded_sequences[i, : len(sequence)] = sequence
        return padded_sequences
    
    def __init__(self, model: str = "GPTCache/albert-duplicate-onnx"):
        tokenizer_name = "albert-base-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = model
        onnx_model_path = hf_hub_download(repo_id=model, filename="model.onnx")
        self.ort_session = ort.InferenceSession(onnx_model_path)

    def eval_single_pair(self, prompt_1: str, prompt_2: str) -> float:
        if prompt_1.lower() == prompt_2.lower():
            return 1.0
        else:
            return self.inference(prompt_1, [prompt_2])

    def eval_one_to_many(self, reference: str, candidates: List[str]) -> List[float]:
        scores = []
        for candidate in candidates:
            score = self.eval_single_pair(reference, candidate)
            scores.append(score)
        return scores

    def inference(self, reference: str, candidates: List[str]) -> np.ndarray:
        """Inference the ONNX model.

        :param reference: reference sentence.
        :type reference: str
        :param candidates: candidate sentences.
        :type candidates: List[str]

        :return: probability score indcates how much is reference similar to candidates.
        """
        n_candidates = len(candidates)
        inference_texts = [
            {"text_a": reference, "text_b": candidate} for candidate in candidates
        ]
        batch_encoding_list = [
            self.tokenizer.encode_plus(
                text["text_a"], text["text_b"], truncation=True, max_length=512, padding="longest"
            )
            for text in inference_texts
        ]

        input_ids_list = [np.array(encode.input_ids) for encode in batch_encoding_list]
        attention_mask_list = [
            np.array(encode.attention_mask) for encode in batch_encoding_list
        ]
        token_type_ids_list = [
            np.array(encode.token_type_ids) for encode in batch_encoding_list
        ]

        padded_input_ids = self.pad_sequence(
            input_ids_list, padding_value=self.tokenizer.pad_token_id
        )
        padded_attention_mask = self.pad_sequence(
            attention_mask_list, padding_value=self.tokenizer.pad_token_id
        )
        padded_token_type_ids = self.pad_sequence(
            token_type_ids_list, padding_value=self.tokenizer.pad_token_id
        )

        ort_inputs = {
            "input_ids": padded_input_ids.reshape(n_candidates, -1),
            "attention_mask": padded_attention_mask.reshape(n_candidates, -1),
            "token_type_ids": padded_token_type_ids.reshape(n_candidates, -1),
        }
        ort_outputs = self.ort_session.run(None, ort_inputs)
        scores = ort_outputs[0][:, 1]
        return float(scores[0])

if __name__ == "__main__":
    evaluator = Onnx_eval()
    prompt_1 = "What is the capital of France?"
    prompt_2 = "What is the capital of France at now?"
    score = evaluator.eval_single_pair(prompt_1, prompt_2)
    