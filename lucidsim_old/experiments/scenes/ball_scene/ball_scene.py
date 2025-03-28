from cmx import doc

doc @ """
# Bouncy Ball

![Generated Image](render/depth.jpg)

This example shows you how to build a simple ball environment, and after that, generate an image out of it.

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

    from vuer import Vuer
    from vuer.schemas import Plane, Sphere, Box, group, GrabRender, AmbientLight


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


    app = Vuer(queries=dict(grid=False), uri="ws://localhost:8013")

    radius = 10


    # use `start=True` to start the app immediately
    @app.spawn(start=True)
    async def main(session):
        session.upsert @ group(key="scene-group")
        session.upsert @ GrabRender(key="DEFAULT")  # need to add this component to the scene.
        session.upsert @ AmbientLight(color="#fff", intensity=10, key="default-light")


        await sleep(1)

        while True:
            session.update @ group(
                BgSphere(color="black"),
                # Box(args=[15, 15, 15], position=[4, 0.5, 0], materialType="depth", key="room", material=dict(side=1)),
                Sphere(args=[0.5, 20, 20], position=[0, 1, 0], materialType="depth", key="ball"),
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
                Box(args=[15, 15, 15], position=[4, 0.5, 0], materialType="depth", key="room", material=dict(side=1)),
                Plane(args=[100, 100], rotation=[-3.14 / 2, 0, 0], materialType="depth"),
                key='scene-group'
            )
            await sleep(0.05)
            res = await session.grab_render(quality=0.95, downsample=2)
            save_buffer(res.value['frame'], "render/depth-room.jpg")

            img = Image.open(BytesIO(res.value['frame']))
            img_np = np.array(img)
            img_np -= img_np.min(axis=(0, 1))
            img_np = img_np.astype(float) / img_np.max(axis=(0, 1))
            img = Image.fromarray((img_np * 255).astype(np.uint8))

            img.save("render/depth-room.png")
            input("press enter to move to the next scene...")

            session.update @ group(
                BgSphere(color="black"),
                Sphere(args=[0.6, 20, 20], position=[0, 1, 0], materialType="basic", material=dict(color="#fff"), key="ball-black"),
                key='scene-group'
            )
            await sleep(0.05)
            res = await session.grab_render(quality=0.95, downsample=2)
            save_buffer(res.value['frame'], "render/mask.jpg")

            input("press enter to move to the next scene...")
            session.update @ group(
                BgSphere(color="white"),
                Sphere(args=[0.55, 20, 20], position=[0, 1, 0], materialType="basic", material=dict(color="#000"), key="ball-white"),
                key='scene-group'
            )
            await sleep(0.05)
            res = await session.grab_render(quality=0.95, downsample=2)
            save_buffer(res.value['frame'], "render/background.jpg")

            input("press enter to move to the next scene...")

doc @ """
"""

doc.flush()
