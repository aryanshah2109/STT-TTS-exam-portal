import asyncio
import time
from typing import List

class BatchRequest:
    def __init__(self, payload, future):
        self.payload = payload
        self.future = future


class AsyncBatcher:
    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_time: float = 0.5
    ):
        self.queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time

    async def add(self, payload):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put(BatchRequest(payload, future))
        return await future

    async def run(self, process_fn):
        while True:
            batch: List[BatchRequest] = []
            start = time.time()

            while len(batch) < self.max_batch_size:
                timeout = self.max_wait_time - (time.time() - start)
                if timeout <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            if not batch:
                continue

            payloads = [b.payload for b in batch]
            results = await process_fn(payloads)

            for req, res in zip(batch, results):
                req.future.set_result(res)
