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

    from vuer import Vuer, VuerSession
    from vuer.schemas import Scene, Plane, Sphere, group, GrabRender, AmbientLight, Box, DirectionalLight, SpotLight, PointLight


    class BgSphere(Sphere):

        def __init__(self, color=None, materialType=None, material=None):
            super().__init__()

            self.key = "background-sphere"
            self.materialType = "phong"
            self.args = [30, 20, 20]
            self.position = [0, 0, 0]
            self.materialType = materialType or "basic"
            self.material = material or dict(color=color, side=2)


    def save_buffer(buffer, path):
        with open(path, "wb") as f:
            f.write(buffer)
        print(f"image saved {path}")


    app = Vuer(port=8013, queries=dict(grid=False))

    radius = 10


    # use `start=True` to start the app immediately
    @app.spawn(start=True)
    async def main(session: VuerSession):
        session.upsert @ group(key="scene-group")
        session.upsert @ GrabRender(key="DEFAULT")  # need to add this component to the scene.
        session.upsert @ AmbientLight(color="#fff", intensity=0.4, key="default-light")

        await sleep(1)

        while True:
            session.update @ group(
                BgSphere(color="black"),
                Box(args=[15, 15, 15], position=[0, 0.5, 0], materialType="depth", key="room", material=dict(side=1)),
                Box(args=[3, 0.15, 2], position=[0, 1, -2.5], materialType="depth", key="hurdle-1"),
                Box(args=[3, 0.15, 2], position=[0, 1, 0], materialType="depth", key="hurdle-2"),
                Box(args=[3, 0.15, 2], position=[0, 1, 2.5], materialType="depth", key="hurdle-3"),
                Plane(args=[100, 100], rotation=[-3.14 / 2, 0, 0], materialType="depth", key="floor"),
                key='scene-group'
            )
            await sleep(0.1)
            res = await session.grab_render(quality=0.95, downsample=2)
            save_buffer(res.value['frame'], "render/depth.jpg")
            input("press enter to move to the next scene...")

            # session.upsert @ PointLight(color="purple", intensity=10, position=[1, 2, -1], key="ceiling-light-1")
            # session.upsert @ PointLight(color="blue", intensity=5, position=[-2, 4, -3], rotation=[0.2, 0.4, -0.4], key="ceiling-light-2")
            session.update @ group(
                BgSphere(materialType="physical",
                         material=dict(color="#b1b1c2", roughness=0.01, emissive="#222", side=2)),
                PointLight(color="white", intensity=10, position=[1, 3.5, -1]),
                Box(args=[3, 0.15, 2], position=[0, 1, -2.5], materialType="physical",
                    material=dict(color="#b1b1c2", roughness=0.01, emissive="#222"), key="hurdle-1"),
                Box(args=[3, 0.15, 2], position=[0, 1, 0], materialType="physical",
                    material=dict(color="#b1b1c2", roughness=0.01, emissive="#222"), key="hurdle-2"),
                Box(args=[3, 0.15, 2], position=[0, 1, 2.5], materialType="physical",
                    material=dict(color="#b1b1c2", roughness=0.01, emissive="#222"), key="hurdle-3"),
                # this color is the IsaacGym color
                Plane(args=[100, 100], rotation=[-3.14 / 2, 0, 0], materialType="physical",
                      material=dict(color="#b1b1c2", roughness=0.01, emissive="#222")),
                key='scene-group'
            )
            await sleep(0.1)
            res = await session.grab_render(quality=0.95, downsample=2)
            save_buffer(res.value['frame'], "render/rgb.jpg")

            input("press enter to move to the next scene...")
            session.update @ group(
                BgSphere(color="black"),
                Box(args=[3, 0.15, 2], position=[0, 1, -2.5], materialType="basic",
                    material=dict(color="white"), key="hurdle-1"),
                Box(args=[3, 0.15, 2], position=[0, 1, 0], materialType="basic",
                    material=dict(color="white"), key="hurdle-2"),
                Box(args=[3, 0.15, 2], position=[0, 1, 2.5], materialType="basic",
                    material=dict(color="white"), key="hurdle-3"),
                Plane(args=[100, 100], rotation=[-3.14 / 2, 0, 0], materialType="basic", material=dict(color="black")),
                key='scene-group'
            )
            await sleep(0.05)
            res = await session.grab_render(quality=0.95, downsample=2)
            save_buffer(res.value['frame'], "render/mask.jpg")

            input("press enter to move to the next scene...")
            session.update @ group(
                BgSphere(color="white"),
                Box(args=[3, 0.15, 2], position=[0, 1, -2.5], materialType="basic", material=dict(color="black"), key="hurdle-1"),
                Box(args=[3, 0.15, 2], position=[0, 1, 0], materialType="basic", material=dict(color="black"), key="hurdle-2"),
                Box(args=[3, 0.15, 2], position=[0, 1, 2.5], materialType="basic", material=dict(color="black"), key="hurdle-3"),
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
