"""lm-eval adapter for standard decoding, fixed APD, and TC-APD."""

import accelerate
import torch
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
from tc_apd import DecodeConfig, TokenThresholdTable, decode


def _parse_constraints(text: str, tokenizer) -> dict[int, int]:
    """Parse constraint string like "120:THE|121:ANSWER" into position->token_id dict."""
    constraints: dict[int, int] = {}

    if text is None or text.strip() == "":
        return constraints

    for part in text.split("|"):
        if ":" not in part:
            continue

        pos_str, word = part.split(":", 1)

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
        model_path="",
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.0,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        temperature=0.0,
        remasking="low_confidence",
        device="cuda",
        tokenizer_path="",
        quantized=False,
        **kwargs,
    ):
        """Initialize LLaDA evaluation harness."""
        super().__init__()

        accelerator = accelerate.Accelerator()

        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        quantized = self._as_bool(quantized)
        self.device = torch.device(device)
        model_kwargs = {}

        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})
        elif quantized:
            # Packed GPTQ modules must be dispatched while loading; moving the
            # resulting model with model.to(...) is unsupported.
            model_kwargs.update({"device_map": {"": device}})

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        )
        self.model.eval()

        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f"{self.accelerator.device}")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            if not quantized:
                self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path or model_path,
            trust_remote_code=True,
        )

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0

        self.sampling_eps = 0.0
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.temperature = float(temperature)
        self.remasking = remasking

        # ============================================================
        # Decoding mode args
        # ============================================================

        self.print_detail_log = self._as_bool(
            kwargs.pop(
                "print_detail_log",
                kwargs.pop("print_detail_log ", True),
            )
        )

        # 1. Whether to use adaptive parallel decoding.
        self.use_adaptive_parallel = self._as_bool(
            kwargs.pop("use_adaptive_parallel", False)
        )

        # 2. Whether to use token-wise dynamic threshold decoding.
        # If True, this has priority over use_adaptive_parallel.
        self.use_dynamic_threshold = self._as_bool(
            kwargs.pop("use_dynamic_threshold", False)
        )

        requested_decoder = str(kwargs.pop("decoder", "")).strip().lower()
        aliases = {
            "": "",
            "standard": "standard",
            "baseline": "standard",
            "fixed": "fixed_apd",
            "fixed_apd": "fixed_apd",
            "adaptive_parallel": "fixed_apd",
            "tc_apd": "tc_apd",
            "token_threshold": "tc_apd",
        }
        if requested_decoder not in aliases:
            raise ValueError(
                f"Unsupported decoder={requested_decoder!r}; expected "
                "standard, fixed_apd, or tc_apd"
            )
        if requested_decoder:
            self.decoder = aliases[requested_decoder]
        elif self.use_dynamic_threshold:
            self.decoder = "tc_apd"
        elif self.use_adaptive_parallel:
            self.decoder = "fixed_apd"
        else:
            self.decoder = "standard"

        # Keep the legacy flags synchronized for old integration branches.
        self.use_dynamic_threshold = self.decoder == "tc_apd"
        self.use_adaptive_parallel = self.decoder == "fixed_apd"

        # 3. Token threshold JSON path.
        # Expected JSON format:
        #   {"123": 0.85, "456": 0.92}
        self.dynamic_threshold_json = kwargs.pop(
            "threshold_path",
            kwargs.pop("dynamic_threshold_json", ""),
        )

        # Token-wise threshold args.
        self.max_threshold = float(kwargs.pop("max_threshold", 0.9))
        self.min_threshold = float(kwargs.pop("min_threshold", 0.02))
        raw_default_threshold = kwargs.pop("default_threshold", None)
        self.default_threshold = (
            0.9 if raw_default_threshold is None else float(raw_default_threshold)
        )

        # Adaptive parallel args.
        self.confidence_threshold = float(kwargs.pop("confidence_threshold", 0.9))
        self.min_parallel_tokens = int(kwargs.pop("min_parallel_tokens", 1))
        self.max_parallel_tokens = int(kwargs.pop("max_parallel_tokens", 100))

        # Optional constraints, mainly for baseline generate.
        self.constraints_text = kwargs.pop("constraints_text", "")

        self.threshold_table = None
        self.dynamic_threshold_dict = {}
        if self.decoder == "tc_apd":
            if not str(self.dynamic_threshold_json).strip():
                raise ValueError("decoder=tc_apd requires dynamic_threshold_json")
            self.threshold_table = TokenThresholdTable.load(
                self.dynamic_threshold_json,
                default_threshold=(
                    None if raw_default_threshold is None else self.default_threshold
                ),
            )
            self.default_threshold = self.threshold_table.default_threshold
            self.dynamic_threshold_dict = dict(self.threshold_table.thresholds)
        self.enable_stagger_aware = self._as_bool(
            kwargs.pop("enable_stagger_aware", False)
        )
        self.enable_soar = self._as_bool(kwargs.pop("enable_soar", False))
        self.max_beam_size = int(kwargs.pop("max_beam_size", 2))
        if self.rank == 0:
            print("====== LLaDA Harness Decoding Config ======")
            print(f"decoder: {self.decoder}")
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
            print(f"temperature: {self.temperature}")
            print(f"print_detail_log: {self.print_detail_log}")
            print("==========================================")

    def _as_bool(self, v, default=False):
        if isinstance(v, bool):
            return v

        if v is None:
            return default

        s = str(v).strip().strip("'\"").lower()
        return s in ("1", "true", "yes", "y", "t")

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        """Identifier used by lm-eval when fingerprinting chat prompts."""
        return str(getattr(self.tokenizer, "name_or_path", "LLaDA-tokenizer"))

    def chat_template(self, chat_template=False):
        """Return the tokenizer's trained chat template for reproducibility."""
        if chat_template is False or chat_template is None:
            return None
        template = getattr(self.tokenizer, "chat_template", None)
        if isinstance(template, dict):
            if isinstance(chat_template, str):
                return template[chat_template]
            if "default" in template:
                return template["default"]
            raise ValueError(
                "Tokenizer has multiple chat templates but no default template"
            )
        return template

    def apply_chat_template(self, chat_history, add_generation_prompt=True) -> str:
        """Format lm-eval chat messages with the model tokenizer."""
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def _forward_process(self, batch, prompt_index):
        batch_size, sequence_length = batch.shape

        target_len = (sequence_length - prompt_index.sum()).item()

        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(
                float(k),
                k + (batch_size - 1) * (target_len / batch_size),
                steps=batch_size,
                device=batch.device,
            )
        ).long()

        x = ((x - 1) % target_len) + 1

        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(batch_size, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(batch_size):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (
                torch.zeros(
                    batch_size,
                    prompt_index.sum(),
                    dtype=torch.bool,
                    device=batch.device,
                ),
                is_mask,
            ),
            dim=1,
        )

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, sequence_length)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.0:
            assert len(prompt_index) == batch.shape[1]

            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)

            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id

            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)

        return logits[:, : batch.shape[1]]

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
                    reduction="none",
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
        seq[0, : len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = seq == self.mask_id

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

        correct = target == seq[0, len(prefix) :]
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
        """
        Memory-friendly generate_until.

        This avoids materializing all requests into a HuggingFace Dataset and avoids
        Dataset.map / Arrow buffers. We stream through lm-eval requests directly.

        Generation logic is kept the same:
        1. Dynamic token-wise threshold decoding
        2. Adaptive parallel decoding
        3. Baseline decoding
        """

        out = []

        for req_idx, req in enumerate(tqdm(requests, desc="Generating...")):
            question_text = req.args[0]
            request_kwargs = req.args[1] if len(req.args) > 1 else {}
            stop_tokens = request_kwargs.get("until", []) or []

            # Tokenize one request at a time to avoid Dataset/Arrow memory overhead.
            input_ids = self.tokenizer(question_text)["input_ids"]
            prompt = torch.tensor(
                input_ids, dtype=torch.long, device=self.device
            ).unsqueeze(0)

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

            if self.decoder == "tc_apd" and self.enable_stagger_aware:
                print("[INFO] Using stagger-aware TC-APD...")
                from generate import generate_token_threshold_parallel_straggler_aware

                generated_out = generate_token_threshold_parallel_straggler_aware(
                    self.model,
                    prompt,
                    threshold_dict=self.dynamic_threshold_dict,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=self.temperature,
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
            elif self.decoder == "tc_apd" and self.enable_soar:
                print("[INFO] Using SOAR with TC-APD thresholds...")
                from generate import generate_soar_token_threshold

                generated_out = generate_soar_token_threshold(
                    self.model,
                    prompt,
                    threshold_dict=self.dynamic_threshold_dict,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=self.temperature,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    log=False,
                    max_threshold=self.max_threshold,
                    min_threshold=self.min_threshold,
                    default_threshold=self.default_threshold,
                    min_parallel_tokens=self.min_parallel_tokens,
                    max_parallel_tokens=self.max_parallel_tokens,
                    max_beam_size=self.max_beam_size,
                )
            else:
                result = decode(
                    self.model,
                    prompt,
                    DecodeConfig(
                        mode=self.decoder,
                        steps=self.steps,
                        gen_length=self.gen_length,
                        block_length=self.block_length,
                        temperature=self.temperature,
                        cfg_scale=self.cfg,
                        remasking=self.remasking,
                        mask_id=self.mask_id,
                        confidence_threshold=self.confidence_threshold,
                        default_threshold=self.default_threshold,
                        min_threshold=self.min_threshold,
                        max_threshold=self.max_threshold,
                        min_parallel_tokens=self.min_parallel_tokens,
                        max_parallel_tokens=self.max_parallel_tokens,
                        collect_traces=self.print_detail_log,
                    ),
                    threshold_table=self.threshold_table,
                    constraints=constraints,
                )
                generated_out = result.sequences
                result.print_report(include_traces=self.print_detail_log)

            generated_answer = self.tokenizer.decode(
                generated_out[0][prompt.shape[1] :],
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
            if not hasattr(self, "_rank") or self.rank == 0:
                print(f"[LOG][Index] {req_idx}")
                print(f"[LOG][Prompt] {question_text}")
                print(f"[LOG][Answer] {generated_answer}\n")

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
