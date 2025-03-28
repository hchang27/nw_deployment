from asyncio import sleep

from vuer import Vuer
from vuer.schemas import DefaultScene, CameraView, Sphere

from lucidsim_old.job_queue import JobQueue

app = Vuer(queries=dict(grid=False))

virtual_camera = CameraView(key="ego", stream="ondemand", monitor=False)
scene = DefaultScene(rawChildren=[virtual_camera])

job_queue = JobQueue()
for i in range(100):
    job_queue.append({"param_1": i * 100, "param_2": f"key-{i}"})

results = []


@app.spawn(start=True)
async def show_heatmap(proxy):
    proxy.set @ scene
    await sleep(0.0)

    # here is the job handling logic: might want to switch to a context manager.

    while True:
        print("jobs left:", len(job_queue))
        print("total results", len(results))
        key, mark_done, put_back = job_queue.take()
        try:
            print(f"I took job-{key}.")
            # print(""" Put your work over here. """)
            for step in range(100):
                # update scene with params:
                proxy.upsert @ Sphere(args=[1, 10, 10], position=[0, 0, step], key="sphere")
                await sleep(0.02)
                # uncomment the following line to grab the rendering result.
                result = await proxy.grab_render(downsample=1, key="ego")
            
            results.append(result)

            # print("Job is completed.")
            # now the job has been finished, we mark it as done by removing it from the queue.
            mark_done()
        except:
            print("Oops, something went wrong. Putting the job back to the queue.")
            put_back()
