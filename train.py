import json
import logging
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

_save_pool = ThreadPoolExecutor(max_workers=1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
DATA_FILE = "data/paired/lgbt/training_data.jsonl"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LORA_RANK = 32
MAX_LENGTH = 4096
EVAL_EVERY = 50
EVAL_SIZE = 200
SEED = 42

SYSTEM_PROMPT = (
    "You are an anonymous poster on /lgbt/. "
    "Reply to the post naturally, matching the tone and style of the board."
)


def load_data(filepath: str) -> list[list[dict]]:
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            data.append(entry["messages"])
    return data


def main():
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"LoRA rank: {LORA_RANK}, LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}")

    all_messages = load_data(DATA_FILE)
    logger.info(f"Loaded {len(all_messages)} training examples")

    random.seed(SEED)
    random.shuffle(all_messages)
    eval_data = all_messages[:EVAL_SIZE]
    train_data = all_messages[EVAL_SIZE:]
    logger.info(f"Train: {len(train_data)}, Eval: {len(eval_data)}")

    tokenizer = get_tokenizer(MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Renderer: {renderer_name}")

    logger.info("Connecting to Tinker...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=MODEL_NAME,
        rank=LORA_RANK,
    )
    logger.info("Connected!")

    logger.info("Preparing eval datums...")
    eval_datums = []
    for messages in eval_data:
        try:
            datum = conversation_to_datum(
                messages, renderer, MAX_LENGTH, renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE
            )
            eval_datums.append(datum)
        except Exception:
            continue
    logger.info(f"Eval datums: {len(eval_datums)}")

    n_batches = len(train_data) // BATCH_SIZE
    logger.info(f"Training for {n_batches} steps (1 epoch)")

    random.shuffle(train_data)
    step = 0
    best_eval_loss = float("inf")
    pending_save: Future | None = None

    for batch_idx in range(n_batches):
        start_time = time.time()
        step = batch_idx + 1

        lr_mult = max(0.0, 1.0 - batch_idx / n_batches)
        current_lr = LEARNING_RATE * lr_mult
        adam_params = tinker.AdamParams(
            learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8
        )

        batch_start = batch_idx * BATCH_SIZE
        batch_messages = train_data[batch_start : batch_start + BATCH_SIZE]

        batch_datums = []
        for messages in batch_messages:
            try:
                datum = conversation_to_datum(
                    messages, renderer, MAX_LENGTH, renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE
                )
                batch_datums.append(datum)
            except Exception:
                continue

        if len(batch_datums) < 2:
            continue

        fwd_bwd_future = training_client.forward_backward(batch_datums, loss_fn="cross_entropy")
        optim_future = training_client.optim_step(adam_params)

        fwd_bwd_result = fwd_bwd_future.result()
        optim_result = optim_future.result()

        train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
        train_weights = [d.loss_fn_inputs["weights"] for d in batch_datums]
        train_nll = compute_mean_nll(train_logprobs, train_weights)

        elapsed = time.time() - start_time
        n_tokens = sum(d.model_input.length for d in batch_datums)

        if step % 5 == 0 or step == 1:
            logger.info(
                f"Step {step}/{n_batches} | Loss: {train_nll:.4f} | "
                f"LR: {current_lr:.2e} | {n_tokens} tok | {elapsed:.1f}s"
            )

        if pending_save is not None and pending_save.done():
            try:
                path = pending_save.result()
                logger.info(f"  >> Checkpoint saved: {path}")
            except Exception as e:
                logger.warning(f"  >> Checkpoint save failed: {e}")
            pending_save = None

        if step % EVAL_EVERY == 0:
            eval_result = training_client.forward_backward(eval_datums, loss_fn="cross_entropy").result()
            eval_logprobs = [x["logprobs"] for x in eval_result.loss_fn_outputs]
            eval_weights = [d.loss_fn_inputs["weights"] for d in eval_datums]
            eval_nll = compute_mean_nll(eval_logprobs, eval_weights)
            logger.info(f"  >> Eval loss: {eval_nll:.4f}")

            if eval_nll < best_eval_loss:
                best_eval_loss = eval_nll
                if pending_save is not None:
                    pending_save.result()
                    pending_save = None
                save_name = f"lgbtgpt-best-step{step}"
                logger.info(f"  >> New best! Saving as {save_name} in background...")
                def _save_best(tc=training_client, name=save_name):
                    return tc.save_weights_for_sampler(name=name).result().path
                pending_save = _save_pool.submit(_save_best)

    logger.info("Training complete!")
    final_name = f"lgbtgpt-final-step{step}"
    final_path = training_client.save_weights_for_sampler(name=final_name).result().path
    logger.info(f"Final model: {final_path}")

    logger.info("Testing model...")
    sampling_client = training_client.create_sampling_client(model_path=final_path)

    test_posts = [
        "how do i know if im actually trans or just a confused femboy",
        ">be me\n>mtf, 6 months hrt\n>voice training going nowhere\n>sound like a dude on the phone every time\nis it over?",
        "i came out to my parents today and they said they still love me but i can tell theyre disappointed",
    ]

    for post in test_posts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": post},
        ]
        prompt = renderer.build_generation_prompt(messages)
        stop_seqs = renderer.get_stop_sequences()

        result = sampling_client.sample(
            prompt=prompt,
            sampling_params=types.SamplingParams(
                max_tokens=200,
                temperature=0.7,
                stop=stop_seqs,
            ),
            num_samples=3,
        ).result()

        print(f"\nPost: {post}")
        for i, seq in enumerate(result.sequences):
            reply = tokenizer.decode(seq.tokens, skip_special_tokens=True).strip()
            print(f"  Reply {i+1}: {reply}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
