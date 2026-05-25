'''LLaDA evaluation harness with adaptive parallel / token-threshold decoding support.'''

import accelerate
import torch
import re
import json
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel


def _parse_constraints(text: str, tokenizer) -> dict[int, int]:
    """Parse constraint string like "120:THE|121:ANSWER" into position->token_id dict."""
    constraints: dict[int, int] = {}

    if text is None or text.strip() == "":
        return constraints

    for part in text.split('|'):
        if ':' not in part:
            continue

        pos_str, word = part.split(':', 1)

        try:
            pos = int(pos_str.strip())
        except ValueError:
            continue

        word = word.strip()

        # Prepend space for tokenization consistency.
        ids = tokenizer.encode(" " + word, add_special_tokens=False)

        for i, tid in enumerate(ids):
            constraints[pos + i] = tid

    return constraints


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        **kwargs,
    ):
        '''Initialize LLaDA evaluation harness.'''
        super().__init__()

        accelerator = accelerate.Accelerator()

        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}

        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
        self.model.eval()

        self.device = torch.device(device)

        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0

        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking

        # ============================================================
        # Decoding mode args
        # ============================================================

        # 1. Whether to use adaptive parallel decoding.
        self.use_adaptive_parallel = self._as_bool(
            kwargs.pop('use_adaptive_parallel', False)
        )

        # 2. Whether to use token-wise dynamic threshold decoding.
        # If True, this has priority over use_adaptive_parallel.
        self.use_dynamic_threshold = self._as_bool(
            kwargs.pop('use_dynamic_threshold', False)
        )

        # 3. Token threshold JSON path.
        # Expected JSON format:
        #   {"123": 0.85, "456": 0.92}
        self.dynamic_threshold_json = kwargs.pop('dynamic_threshold_json', '')

        # Token-wise threshold args.
        self.max_threshold = float(kwargs.pop('max_threshold', 0.9))
        self.min_threshold = float(kwargs.pop('min_threshold', 0.02))
        self.default_threshold = float(kwargs.pop('default_threshold', 0.9))

        # Adaptive parallel args.
        self.confidence_threshold = float(kwargs.pop('confidence_threshold', 0.9))
        self.min_parallel_tokens = int(kwargs.pop('min_parallel_tokens', 1))
        self.max_parallel_tokens = int(kwargs.pop('max_parallel_tokens', 100))

        # Optional constraints, mainly for baseline generate.
        self.constraints_text = kwargs.pop('constraints_text', '')

        # Load dynamic threshold dict once at initialization.
        self.dynamic_threshold_dict = self._load_threshold_dict(
            self.dynamic_threshold_json
        )

        if self.rank == 0:
            print("====== LLaDA Harness Decoding Config ======")
            print(f"use_dynamic_threshold: {self.use_dynamic_threshold}")
            print(f"use_adaptive_parallel: {self.use_adaptive_parallel}")
            print(f"dynamic_threshold_json: {self.dynamic_threshold_json}")
            print(f"loaded token thresholds: {len(self.dynamic_threshold_dict)}")
            print(f"max_threshold: {self.max_threshold}")
            print(f"min_threshold: {self.min_threshold}")
            print(f"default_threshold: {self.default_threshold}")
            print(f"confidence_threshold: {self.confidence_threshold}")
            print(f"min_parallel_tokens: {self.min_parallel_tokens}")
            print(f"max_parallel_tokens: {self.max_parallel_tokens}")
            print("==========================================")

        if self.use_dynamic_threshold and len(self.dynamic_threshold_dict) == 0:
            print(
                "[WARN] use_dynamic_threshold=True but no threshold dict loaded. "
                "All tokens will fallback to default_threshold."
            )

    def _as_bool(self, v, default=False):
        if isinstance(v, bool):
            return v

        if v is None:
            return default

        s = str(v).strip().strip("'\"").lower()
        return s in ("1", "true", "yes", "y", "t")

    def _load_threshold_dict(self, json_path):
        """
        Load token-wise threshold json.

        Expected JSON format:
            {"123": 0.85, "456": 0.92}

        Return:
            {123: 0.85, 456: 0.92}
        """
        if json_path is None:
            return {}

        json_path = str(json_path).strip().strip("'\"")

        if json_path == "":
            return {}

        path = Path(json_path)

        if not path.exists():
            print(f"[WARN] dynamic_threshold_json not found: {path}")
            return {}

        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        threshold_dict = {}

        for k, v in raw.items():
            try:
                threshold_dict[int(k)] = float(v)
            except Exception:
                continue

        print(f"[INFO] Loaded {len(threshold_dict)} token thresholds from {path}")

        return threshold_dict

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()

        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(
                float(k),
                k + (b - 1) * (target_len / b),
                steps=b,
                device=batch.device,
            )
        ).long()

        x = ((x - 1) % target_len) + 1

        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (
                torch.zeros(
                    b,
                    prompt_index.sum(),
                    dtype=torch.bool,
                    device=batch.device,
                ),
                is_mask,
            ),
            dim=1,
        )

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]

            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)

            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id

            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)

        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []

        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = (
                F.cross_entropy(
                    logits[mask_indices],
                    seq[mask_indices],
                    reduction='none',
                )
                / p_mask[mask_indices]
            )

            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full(
            (1, len(prefix) + len(target)),
            self.mask_id,
            device=self.device,
        )

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)

            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(
                p,
                dim=-1,
                index=torch.unsqueeze(x0, -1),
            ).squeeze(dim=-1)

            _, index = torch.sort(confidence, descending=True)

            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()

        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)

        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())

        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])

            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]

        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []

        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)
                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))

        torch.cuda.empty_cache()

        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [
            {
                "question": req.args[0],
                "until": req.args[1]['until'],
            }
            for req in requests
        ]

        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []

        for elem in tqdm(ds, desc="Generating..."):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]

            constraints = (
                _parse_constraints(self.constraints_text, self.tokenizer)
                if self.constraints_text
                else None
            )

            # ============================================================
            # Generate:
            # 1. Dynamic token-wise threshold decoding
            # 2. Adaptive parallel decoding
            # 3. Baseline decoding
            # ============================================================

            if self.use_dynamic_threshold:
                from generate import generate_token_threshold_parallel

                generated_out = generate_token_threshold_parallel(
                    self.model,
                    prompt,
                    threshold_dict=self.dynamic_threshold_dict,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    log=False,
                    max_threshold=self.max_threshold,
                    min_threshold=self.min_threshold,
                    default_threshold=self.default_threshold,
                    min_parallel_tokens=self.min_parallel_tokens,
                    max_parallel_tokens=self.max_parallel_tokens,
                )

            elif self.use_adaptive_parallel:
                from generate import generate_adaptive_parallel

                generated_out = generate_adaptive_parallel(
                    self.model,
                    prompt,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    log=False,
                    confidence_threshold=self.confidence_threshold,
                    min_parallel_tokens=self.min_parallel_tokens,
                    max_parallel_tokens=self.max_parallel_tokens,
                )

            else:
                from generate import generate as generate_baseline

                generated_out = generate_baseline(
                    self.model,
                    prompt,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    constraints=constraints,
                )

            generated_answer = self.tokenizer.decode(
                generated_out[0][prompt.shape[1]:],
                skip_special_tokens=False,
            )

            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            # Remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids,
                skip_special_tokens=True,
            )

            out.append(generated_answer)

            # Log input & output
            if not hasattr(self, '_rank') or self.rank == 0:
                question_text = elem.get("question_text", "<N/A>")
                print(f"[LOG][Prompt] {question_text}")
                print(f"[LOG][Answer] {generated_answer}\n")

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()