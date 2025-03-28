from asyncio.exceptions import TimeoutError
from pathlib import Path

from cmx import doc

doc @ """
# Stair Way with a Cone Marker

![Generated Image]()

This example shows you how to build a more complex scene that involves multiple object categories. 

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
    from vuer.schemas import Scene, Plane, Sphere, group, GrabRender, AmbientLight, Box, PointLight, Cone, Cylinder


    class BgSphere(Sphere):

        def __init__(self, color, materialType="phong"):
            super().__init__()

            self.key = "background-sphere"
            self.materialType = materialType
            self.args = [100, 20, 20]
            self.position = [0, 0, 0]
            self.materialType = "basic"
            self.material = dict(color=color, side=2)


    def save_buffer(buffer, path:Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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
        session.upsert @ AmbientLight(color="#fff", intensity=0.4, key="default-light")

        await sleep(1)

        while True:

            session.update @ group(
                BgSphere(color="#b1b1c2", materialType="basic"),
                PointLight(color="white", intensity=10, position=[0, 2, 0]),
                PointLight(color="white", intensity=15, position=[-3, 2, 2.5]),
                Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625, 1], key="cone-1", materialType="standard", material=dict(emissive="#222", roughness=0.05, color="#FFA500")),
                Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625 + 0.45, 0], key="cone-2", materialType="standard", material=dict(emissive="#222", roughness=0.05, color="#FFA500")),
                Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625 + 0.9, -1], key="cone-3", materialType="standard", material=dict(emissive="#222", roughness=0.05, color="#FFA500")),
                Box(args=[3, 0.7, 1], position=[0, 0.15, 1], key="step-1", materialType="standard",
                    material=dict(emissive="#222", roughness=0.01, color="#b1b1c2")),
                Box(args=[3, 0.7, 1], position=[0, 0.6, 0], key="step-2", materialType="standard",
                    material=dict(emissive="#222", roughness=0.01, color="#b1b1c2")),
                Box(args=[3, 0.7, 1], position=[0, 1.1, -1], key="step-3", materialType="standard",
                    material=dict(emissive="#222", roughness=0.01, color="#b1b1c2")),
                Plane(args=[100, 100], rotation=[-3.14 / 2, 0, 0], materialType="standard", material=dict(color="#b1b1c2", emissive="#222", roughness=0.1) ),
                key='scene-group'
            )
            await sleep(0.05)
            res = await session.grab_render(quality=1, downsample=2, renderDepth=True)
            save_buffer(res.value['frame'], "render/rgb.jpg")
            save_buffer(res.value['depth'], "render/depth.jpg")

            input("press enter to move to the next scene...")
            session.update @ group(
                BgSphere(color="black"),
                Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625, 1], key="cone-1", materialType="basic", material=dict(color="black")),
                Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625 + 0.45, 0], key="cone-2", materialType="basic", material=dict(color="black")),
                Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625 + 0.9, -1], key="cone-3", materialType="basic", material=dict(color="black")),
                Box(args=[3, 0.7, 1], position=[0, 0.15, 1], key="step-1", materialType="basic", material=dict(color="white")),
                Box(args=[3, 0.7, 1], position=[0, 0.6, 0], key="step-2", materialType="basic", material=dict(color="white")),
                Box(args=[3, 0.7, 1], position=[0, 1.1, -1], key="step-3", materialType="basic", material=dict(color="white")),
                Plane(args=[100, 100], rotation=[-3.14 / 2, 0, 0], materialType="basic", material=dict(color="black")),
                key='scene-group'
            )
            await sleep(0.05)
            res = await session.grab_render(quality=1, downsample=2)
            save_buffer(res.value['frame'], "render/mask.jpg")

            input("press enter to move to the next scene...")
            session.update @ group(
                BgSphere(color="white"),
                Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625, 1], key="cone-1", materialType="basic", material=dict(color="black")),
                Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625 + 0.45, 0], key="cone-2", materialType="basic", material=dict(color="black")),
                Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625 + 0.9, -1], key="cone-3", materialType="basic", material=dict(color="black")),
                Box(args=[3, 0.7, 1], position=[0, 0.15, 1], key="step-1", materialType="basic", material=dict(color="black")),
                Box(args=[3, 0.7, 1], position=[0, 0.6, 0], key="step-2", materialType="basic", material=dict(color="black")),
                Box(args=[3, 0.7, 1], position=[0, 1.1, -1], key="step-3", materialType="basic", material=dict(color="black")),
                Plane(args=[100, 100], rotation=[-3.14 / 2, 0, 0], materialType="basic", material=dict(color="white")),
                key='scene-group'
            )
            await sleep(0.05)
            res = await session.grab_render(quality=1, downsample=2)
            save_buffer(res.value['frame'], "render/background.jpg")

            input("press enter to move to the next scene...")
            session.update @ group(
                BgSphere(color="black"),
                Cylinder(args=[0.06, 0.12, 0.25, 20], position=[0, 0.625, 1], key="cone-1", materialType="basic", material=dict(color="white")),
                Cylinder(args=[0.06, 0.12, 0.25, 20], position=[0, 0.625 + 0.45, 0], key="cone-2", materialType="basic", material=dict(color="white")),
                Cylinder(args=[0.06, 0.12, 0.25, 20], position=[0, 0.625 + 0.9, -1], key="cone-3", materialType="basic", material=dict(color="white")),
                Box(args=[3, 0.7, 1], position=[0, 0.15, 1], key="step-1", materialType="basic", material=dict(color="black")),
                Box(args=[3, 0.7, 1], position=[0, 0.6, 0], key="step-2", materialType="basic", material=dict(color="black")),
                Box(args=[3, 0.7, 1], position=[0, 1.1, -1], key="step-3", materialType="basic", material=dict(color="black")),
                Plane(args=[100, 100], rotation=[-3.14 / 2, 0, 0], materialType="basic", material=dict(color="black")),
                key='scene-group'
            )
            await sleep(0.05)
            res = await session.grab_render(quality=1, downsample=2)
            save_buffer(res.value['frame'], "render/target.jpg")

            input("We are done! Press enter to restart...")
doc @ """
"""

doc.flush()
