# Example for using the camera

```python
import time
import numpy as np
import asyncio
from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, ImageBackground
from vuer.serdes import jpg

from zkit import Camera

zed = Camera(fps=120)
# zed.share_buffers("rgb", "depth")
zed.spin_process("rgb", "depth")
time.sleep(1)

app = Vuer(
    uri="ws://localhost:8112",
    queries=dict(
        reconnect=True,
        grid=False,
        backgroundColor="black",
    ),
    port=8112,
)

@app.spawn(start=True)
async def show_heatmap(session: VuerSession):
    session.set @ DefaultScene()

    def clean_depth(depth, max=1, min=0):
        depth = depth.copy()
        depth = np.nan_to_num(depth)
        depth[depth >= max] = max
        depth[depth <= min] = min
        return depth

    i = 0
    while True:
        rgb, depth = zed.frame.values()

        depth = clean_depth(depth / 2_000)
        depth = (depth  * 255).astype(np.uint8)

        session.upsert(
            ImageBackground(
                # src=jpg(rgb, 99),
                src=jpg(depth, 99),
                depthSrc=jpg(depth),
                key="image",
            ),
            to="bgChildren",
        )
        await asyncio.sleep(0.01)
```