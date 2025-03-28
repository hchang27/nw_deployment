from asyncio.exceptions import TimeoutError

from cmx import doc

doc @ """
# Bouncy Ball

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

    from vuer import Vuer
    from vuer.schemas import Scene, Plane, Sphere, group, GrabRender


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


    app = Vuer(queries=dict(grid=False))

    radius = 10


    # use `start=True` to start the app immediately
    @app.spawn(start=True)
    async def main(session):
        session.set @ Scene(
            group(key="scene-group"),
            GrabRender(key="DEFAULT"), # need to add this component to the scene.
        )
        # await sleep(0.0)

        while True:
            session.update @ group(
                BgSphere(color="black"),
                Sphere(args=[0.5, 20, 20], position=[0, 1, 0], materialType="depth", key="ball"),
                Plane(args=[100, 100], rotation=[-3.14 / 2, 0, 0], materialType="depth"),
                key='scene-group'
            )
            await sleep(0.05)
            res = await session.grab_render(quality=0.95, downsample=2)
            save_buffer(res.value['frame'], "render/depth.jpg")
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
