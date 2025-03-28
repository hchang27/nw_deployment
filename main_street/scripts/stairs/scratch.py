import sys
from asyncio import sleep

import numpy as np
from vuer import Vuer
from vuer.schemas import Sphere, DefaultScene, CameraView
import time

from asyncio.exceptions import TimeoutError

app = Vuer(queries=dict(grid=False))


# We don't auto start the vuer app because we need to bind a handler.
@app.spawn(start=True)
async def show_heatmap(proxy):
    proxy.set @ DefaultScene(
        Sphere(
            key="sphere",
            args=[0.1, 20, 20],
            position=[0, 0, 0],
            rotation=[0, 0, 0],
            materialType="depth",
        ),
        rawChildren=[
            CameraView(
                fov=50,
                width=320,
                height=240,
                key="ego",
                position=[-0.5, 1.25, 0.5],
                rotation=[-0.4 * np.pi, -0.1 * np.pi, 0.15 + np.pi],
                stream="ondemand",
                fps=30,
                near=0.45,
                far=1.8,
                showFrustum=True,
                downsample=1,
                distanceToCamera=2
                # dpr=1,
            ),
        ],
        # hide the helper to only render the objects.
        show_helper=False,
    )
    await sleep(0.0)

    i = 0
    while True:
        # await sleep(0.01) 
        i += 1
        h = 0.25 - (0.00866 * (i % 120 - 60)) ** 2
        position = [0.2, 0.0, 0.1 + h]

        proxy.update @ [
            Sphere(
                key="sphere",
                args=[0.1, 20, 20],
                position=position,
                rotation=[0, 0, 0],
                materialType="depth",
            ),
        ]

        time.sleep(1.0)
        try:
            result = await proxy.grab_render(key="ego", quality=0.9)
            print("you render came back with keys: [")
        except TimeoutError:
            print("timeout")
        finally:
            print("done")


app.run()
