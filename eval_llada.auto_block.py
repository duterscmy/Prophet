'''LLaDA evaluation harness with adaptive auto-block decoding only.'''

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

# If your file is generate/auto_block.py
from generate.auto_block import generate as generate_auto_block

# If your file is generate_auto_block.py instead, use:
# from generate_auto_block import generate as generate_auto_block


def _parse_constraints(text: str, tokenizer) -> dict[int, int]:
    """Parse constraint string like '120:THE|121:ANSWER' into position->token_id dict."""
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
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking="low_confidence",
        device="cuda",
        **kwargs,
    ):
        super().__init__()

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None

        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})

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
            self.device = torch.device(f"{self.accelerator.device}")
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
        self.steps = int(steps)
        self.gen_length = int(gen_length)
        self.block_length = int(block_length)
        self.remasking = remasking

        # Optional constraints.
        self.constraints_text = kwargs.pop("constraints_text", "")

        # =========================================================
        # Auto-block decoding args
        # =========================================================
        self.adaptive_block_size = self._as_bool(
            kwargs.pop("adaptive_block_size", True)
        )

        self.adaptive_min_block_size = int(
            kwargs.pop("adaptive_min_block_size", 2)
        )

        adaptive_max_block_size = kwargs.pop("adaptive_max_block_size", None)
        if (
            adaptive_max_block_size is None
            or str(adaptive_max_block_size).strip().lower() in ("none", "null", "")
        ):
            self.adaptive_max_block_size = None
        else:
            self.adaptive_max_block_size = int(adaptive_max_block_size)

        # Candidate search:
        #   "power2": [2,4,8,16,32,64,128]
        #   "dense":  [2,3,4,...,128]
        self.adaptive_candidate_mode = str(
            kwargs.pop("adaptive_candidate_mode", "dense")
        ).strip()

        self.adaptive_candidate_stride = int(
            kwargs.pop("adaptive_candidate_stride", 1)
        )

        # Optional coarse-to-fine refinement.
        self.adaptive_refine_candidates = self._as_bool(
            kwargs.pop("adaptive_refine_candidates", False)
        )

        self.adaptive_refine_stride = int(
            kwargs.pop("adaptive_refine_stride", 1)
        )

        # Gap score:
        #   window: G(B)=mean(c_{B-w:B}) - mean(c_{B:B+w})
        #   block:  G(B)=mean(c_{0:B})   - mean(c_{B:2B})
        self.gap_context_mode = str(
            kwargs.pop("gap_context_mode", "block")
        ).strip()

        self.use_gap_score = self._as_bool(
            kwargs.pop("use_gap_score", True)
        )

        self.use_length_compensation = self._as_bool(
            kwargs.pop("use_length_compensation", False)
        )

        self.positive_gap_only = self._as_bool(
            kwargs.pop("positive_gap_only", True)
        )

        gap_window_size = kwargs.pop("gap_window_size", None)
        if (
            gap_window_size is None
            or str(gap_window_size).strip().lower() in ("none", "null", "")
        ):
            self.gap_window_size = None
        else:
            self.gap_window_size = int(gap_window_size)

        # Logging.
        self.auto_block_return_logs = self._as_bool(
            kwargs.pop("auto_block_return_logs", True)
        )

        self.auto_block_log = self._as_bool(
            kwargs.pop("auto_block_log", False)
        )

        self.log_all_probe_positions = self._as_bool(
            kwargs.pop("log_all_probe_positions", True)
        )

        self.print_probe_positions = self._as_bool(
            kwargs.pop("print_probe_positions", False)
        )

        self.dump_json_logs = self._as_bool(
            kwargs.pop("dump_json_logs", False)
        )

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
        seq[0, :len(prefix)] = prefix

        for _ in range(len(target)):
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
                "until": req.args[1]["until"],
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

            generated = generate_auto_block(
                self.model,
                prompt,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0.,
                cfg_scale=self.cfg,
                remasking=self.remasking,
                mask_id=self.mask_id,
                constraints=constraints,

                # Adaptive block-size decoding
                adaptive_block_size=self.adaptive_block_size,
                adaptive_min_block_size=self.adaptive_min_block_size,
                adaptive_max_block_size=self.adaptive_max_block_size,

                # Candidate search
                adaptive_candidate_mode=self.adaptive_candidate_mode,
                adaptive_candidate_stride=self.adaptive_candidate_stride,

                # Fine-grained refinement
                adaptive_refine_candidates=self.adaptive_refine_candidates,
                adaptive_refine_stride=self.adaptive_refine_stride,

                # Gap score
                gap_context_mode=self.gap_context_mode,
                use_gap_score=self.use_gap_score,
                use_length_compensation=self.use_length_compensation,
                positive_gap_only=self.positive_gap_only,
                gap_window_size=self.gap_window_size,

                # Logs
                tokenizer=self.tokenizer,
                return_logs=self.auto_block_return_logs,
                log=self.auto_block_log,
                log_all_probe_positions=self.log_all_probe_positions,
                print_probe_positions=self.print_probe_positions,
                dump_json_logs=self.dump_json_logs,
            )

            if self.auto_block_return_logs:
                generated_out, auto_block_logs = generated
            else:
                generated_out = generated
                auto_block_logs = None

            if (
                (not hasattr(self, "_rank") or self.rank == 0)
                and auto_block_logs is not None
            ):
                summary = auto_block_logs.get("summary", {})
                print("[AUTO_BLOCK][Summary]", summary)

            generated_answer = self.tokenizer.decode(
                generated_out[0][prompt.shape[1]:],
                skip_special_tokens=False,
            )

            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids,
                skip_special_tokens=True,
            )

            out.append(generated_answer)

            if not hasattr(self, "_rank") or self.rank == 0:
                question_text = elem.get("question_text", "<N/A>")
                print(f"[LOG][Prompt] {question_text}")
                print(f"[LOG][Answer] {generated_answer}\n")

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()