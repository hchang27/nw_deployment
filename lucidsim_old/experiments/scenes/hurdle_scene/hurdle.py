from asyncio.exceptions import TimeoutError

from cmx import doc

doc @ """
# Hurdle Scene

![Generated Image]()

This example shows you how to build a simple curb, and after that, generate an image out of it.

Make an asset folder for the generated trimesh.
```shell
mkdir -p render
```

Now, run
"""

# with doc, doc.skip:
with doc:
    from asyncio import sleep
    from PIL import Image
    import numpy as np
    from io import BytesIO

    from vuer import Vuer, VuerSession
    from vuer.schemas import Scene, Plane, Sphere, group, GrabRender, AmbientLight, Box


    class BgSphere(Sphere):

        def __init__(self, color):
            super().__init__()

            self.key = "background-sphere"
            self.materialType = "phong"
            self.args = [100, 20, 20]
            self.position = [0, 0, 0]
            self.materialType = "basic"
            self.material = dict(color=color, side=2)


    def save_buffer(buffer, path):
        with open(path, "wb") as f:
            f.write(buffer)
        print(f"image saved {path}")


    app = Vuer(port=8013, uri="ws://localhost:8013", queries=dict(grid=False))

    radius = 10


    # use `start=True` to start the app immediately
    @app.spawn(start=True)
    async def main(session: VuerSession):
        session.upsert @ group(key="scene-group")
        session.upsert @ GrabRender(key="DEFAULT")  # need to add this component to the scene.
        session.upsert @ AmbientLight(color="#fff", intensity=10, key="default-light")

        await sleep(1)

        while True:
            session.update @ group(
                BgSphere(color="black"),
                Box(args=[15, 15, 15], position=[0, 0.5, 0], materialType="depth", key="room", material=dict(side=1)),
                Box(args=[15, 1, 1], position=[0, 0.5, 0], materialType="depth", key="hurdle"),
                Plane(args=[100, 100], rotation=[-3.14 / 2, 0, 0], materialType="depth"),
                key='scene-group'
            )
            await sleep(0.05)
            res = await session.grab_render(quality=0.95, downsample=2)
            save_buffer(res.value['frame'], "render/depth.jpg")

            img = Image.open(BytesIO(res.value['frame']))
            img_np = np.array(img)
            img_np -= img_np.min(axis=(0, 1))
            img_np = img_np.astype(float) / img_np.max(axis=(0, 1))
            img = Image.fromarray((img_np * 255).astype(np.uint8))

            img.save("render/depth.png")
            input("press enter to move to the next scene...")

            session.update @ group(
                BgSphere(color="black"),
                Box(args=[15, 1.2, 1.2], position=[0, 0.5, 0], materialType="basic", key="hurdle-white", material=dict(color="white")),
                Plane(args=[100, 100], rotation=[-3.14 / 2, 0, 0], materialType="basic", material=dict(color="black")),
                key='scene-group'
            )
            await sleep(0.05)
            res = await session.grab_render(quality=0.95, downsample=2)
            save_buffer(res.value['frame'], "render/mask.jpg")

            input("press enter to move to the next scene...")
            session.update @ group(
                BgSphere(color="white"),
                Box(args=[15, 1.05, 1.05], position=[0, 0.5, 0], materialType="basic", key="hurdle-black", material=dict(color="black")),
                Plane(args=[100, 100], rotation=[-3.14 / 2, 0, 0], materialType="basic", material=dict(color="white")),
                key='scene-group'
            )
            await sleep(0.05)
            res = await session.grab_render(quality=0.95, downsample=2)
            save_buffer(res.value['frame'], "render/background.jpg")

            input("press enter to move to the next scene...")

doc @ """
"""

doc.flush()
