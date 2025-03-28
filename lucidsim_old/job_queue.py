from time import time
from uuid import uuid4
from functools import partial
from copy import copy


class JobQueue:
    def __init__(self, jobs=None, ttl=5):
        """A simple job queue.

        Args:
            data (dict): a dictionary of jobs.
            ttl (int, optional): time to live. Defaults to 5.
        """
        self._ttl = ttl
        self.queue = dict()

        if jobs is not None:
            for j in jobs:
                self.append(j)


    def take(self):
        """Grab a job that has not been grabbed from the queue."""

        for k in sorted(self.queue.keys()):
            job = self.queue[k]
            if job["status"] is None:
                job["grab_ts"] = time()
                job["status"] = "in_progress"

                return job, partial(self.mark_done, k), partial(self.mark_reset, k)

        return None, None, None

    def append(self, job_params):
        """Append a job to the queue."""

        k = str(uuid4())
        self.queue[k] = {
            "created_ts": time(),
            "status": None,
            "grab_ts": None,
            "job_params": job_params,
        }

    def __len__(self):
        return len(self.queue)

    def mark_done(self, key):
        """Mark a job as done."""
        del self.queue[key]

    def mark_reset(self, key):
        self.queue[key]["status"] = None
        self.queue[key]["grab_ts"] = None

    def house_keeping(self):
        """Reset jobs that have become stale."""
        for job in self.queue.values():
            if job["status"] or job["grab_ts"] is None:
                continue
            if job["grab_ts"] < (time() - self._ttl):
                job["status"] = None
                del job["grab_ts"]

if __name__ == '__main__':
    queue = JobQueue([{"seed": 1}, {"seed": 2}, {"seed": 3}])
    while queue:
        j, d, r = queue.take()
        print(j)
        d()