"""Canary uses exactly the production collection and scoring pipeline."""
from melchior.crucible.runner import CrucibleRunner


class CanaryRunner(CrucibleRunner):
    def __init__(self, num_tasks=50, concurrency=10, llm_url='http://localhost:8000/v1',
                 llm_model='qwen', jev_url='http://localhost:8080/v1/systemone',
                 output_dir='data/crucible_canary', sandbox_timeout_s=14., min_delta=0.01,
                 *, seed_start=1000, resume=False, max_hours=None):
        super().__init__(concurrency=concurrency, output_dir=output_dir,
                         llm_base_url=llm_url, llm_model=llm_model, llm_mode='api',
                         sandbox_timeout_s=sandbox_timeout_s, max_pairs=num_tasks,
                         split='eval', seed_start=seed_start, resume=resume,
                         max_hours=max_hours, min_delta=min_delta, jev_url=jev_url)
