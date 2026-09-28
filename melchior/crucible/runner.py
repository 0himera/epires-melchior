"""Crucible high-throughput async runner for 20-32 concurrent workers."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from melchior.crucible.environments import generate_task
from melchior.crucible.client import CrucibleLLMClient
from melchior.crucible.sandbox import AsyncSandbox
from melchior.crucible.arbiter import CrucibleArbiter


class CrucibleRunner:
    def __init__(
        self,
        concurrency: int = 20,
        output_dir: str | Path = "data/crucible",
        llm_base_url: str | None = None,
        llm_model: str = "qwen",
        llm_mode: str = "auto",
        sandbox_timeout_s: float = 12.0,
        max_pairs: int | None = None,
    ):
        self.concurrency = concurrency
        self.output_dir = Path(output_dir)
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.llm_mode = llm_mode
        self.sandbox_timeout_s = sandbox_timeout_s
        self.max_pairs = max_pairs

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nli_file = self.output_dir / "openjev_ml_nli.jsonl"
        self.dpo_file = self.output_dir / "melchior_dpo_pairs.jsonl"

        self.arbiter = CrucibleArbiter(min_delta=0.005)
        self.sandbox = AsyncSandbox(timeout_s=self.sandbox_timeout_s)
        self.client = CrucibleLLMClient(
            base_url=self.llm_base_url,
            model=self.llm_model,
            mode=self.llm_mode,
        )

        self._lock = asyncio.Lock()
        self._total_evaluated = 0
        self._nli_rows_written = 0
        self._dpo_rows_written = 0
        self._start_time = 0.0

    async def _worker(self, worker_id: int, queue: asyncio.Queue[int]):
        """Individual worker loop processing tasks."""
        while True:
            seed = await queue.get()
            if seed is None:
                queue.task_done()
                break

            try:
                # 1. Generate empirical task and data split
                profile, X_tr, y_tr, X_va, y_va = generate_task(seed)

                # 2. Get competing candidates from LLM
                pair = await self.client.generate_pair(profile, seed)

                # 3. Execute both candidates in parallel in isolated sandboxes
                res_a, res_b = await asyncio.gather(
                    self.sandbox.execute(pair.code_a, X_tr, y_tr, X_va, y_va),
                    self.sandbox.execute(pair.code_b, X_tr, y_tr, X_va, y_va),
                )

                # 4. Objective empirical judgment
                outcome = self.arbiter.evaluate(profile, pair, res_a, res_b)

                # 5. Thread-safe write to dataset files
                async with self._lock:
                    self._total_evaluated += 1

                    if outcome.nli_records:
                        with open(self.nli_file, "a", encoding="utf-8") as f:
                            for row in outcome.nli_records:
                                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                                self._nli_rows_written += 1

                    if outcome.dpo_record:
                        with open(self.dpo_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(outcome.dpo_record, ensure_ascii=False) + "\n")
                            self._dpo_rows_written += 1

                    # Log progress every 5 evaluations
                    if self._total_evaluated % 5 == 0 or self._total_evaluated == 1:
                        elapsed = time.time() - self._start_time
                        rate = (self._total_evaluated / elapsed) * 60.0 if elapsed > 0 else 0.0
                        winner_label = f"Winner: {outcome.winner} (Δ={outcome.delta:.3f})"
                        print(
                            f"[{self._total_evaluated:05d} pairs] "
                            f"{rate:5.1f} pairs/min | "
                            f"NLI rows: {self._nli_rows_written} | "
                            f"DPO rows: {self._dpo_rows_written} | "
                            f"Last: {winner_label}"
                        )

            except Exception as e:
                # Avoid crash of worker on single task failure
                pass
            finally:
                queue.task_done()

    async def run(self):
        """Starts the async generation crucible."""
        self._start_time = time.time()
        print(f"[*] Melchior Crucible initialized with {self.concurrency} async workers")
        print(f"[*] Output directory: {self.output_dir.resolve()}")
        print(f"[*] Target NLI file:  {self.nli_file.name}")
        print(f"[*] Target DPO file:  {self.dpo_file.name}")
        print(f"[*] LLM backend:      {self.llm_base_url or 'mock/synthetic'} (model: {self.llm_model})")
        print("[*] Starting self-play feedback loop...\n")

        queue: asyncio.Queue[int | None] = asyncio.Queue(maxsize=self.concurrency * 2)

        # Launch workers
        workers = [
            asyncio.create_task(self._worker(i, queue))
            for i in range(self.concurrency)
        ]

        # Producer feed
        seed = 1000
        try:
            while True:
                if self.max_pairs is not None and self._total_evaluated >= self.max_pairs:
                    break
                await queue.put(seed)
                seed += 1
                await asyncio.sleep(0.01)

            # Signal termination
            for _ in range(self.concurrency):
                await queue.put(None)

            await asyncio.gather(*workers)

        except (asyncio.CancelledError, KeyboardInterrupt):
            print("\n[!] Gracefully stopping workers and saving flushed state...")
            for _ in range(self.concurrency):
                try:
                    queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass
        finally:
            await self.client.close()
            elapsed = time.time() - self._start_time
            print(f"\n[✓] Crucible finished in {elapsed:.1f}s.")
            print(f"[✓] Total empirical evaluations: {self._total_evaluated}")
            print(f"[✓] Generated NLI rows:          {self._nli_rows_written} -> {self.nli_file}")
            print(f"[✓] Generated DPO pairs:         {self._dpo_rows_written} -> {self.dpo_file}")
