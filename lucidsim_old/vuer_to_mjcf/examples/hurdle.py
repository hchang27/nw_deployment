import asyncio

import numpy as np
from vuer import Vuer
from vuer.schemas import Box, CameraView, Cylinder, Plane, Scene, group

scene = Scene(group(group(
    Box(args=[1.0, 0.25, 0.25], position=[0, 0, 0.125], rotation=[0, 0, -1.57], key="hurdle1",
        material=dict(color="blue")),
    Box(args=[1.0, 0.25, 0.25], position=[2.0, 0, 0.125], rotation=[0, 0, -1.57], key="hurdle2",
        material=dict(color="blue")),
    Box(args=[1.0, 0.25, 0.25], position=[4.0, 0, 0.125], rotation=[0, 0, -1.57], key="hurdle3",
        material=dict(color="blue")), position=[2.0, 0, 0]),
    Plane(args=[100, 100], material=dict(color="red")), key="scene-group-2"),
    Cylinder(args=[0.5], position=[2, 0, 0], key="cone-1", materialType="standard"),
    Cylinder(args=[0.5], position=[4, 0, 0], key="cone-2", materialType="standard"),
    Cylinder(args=[0.5], position=[6, 0, 0], key="cone-3", materialType="standard"),
    key='scene-group',
    up=[0, 0, 1],
    rawChildren=[CameraView(
        ctype="orthographic",
        fov=50,
        width=320,
        height=240,
        key="ego",
        position=[0, 0, 2],
        rotation=[0, 0, - 0.5 * np.pi],
        stream="ondemand",
        fps=30,
        near=0.45,
        far=2.2,
        renderDepth=True,
        showFrustum=True,
        downsample=1,
        distanceToCamera=2
    )])

if __name__ == '__main__':
    app = Vuer()


    @app.spawn
    async def main(session):
        session.set @ scene

        # result = await session.grab_render(downsample=1, key="ego")
        # 
        # frame = result.value["depthFrame"] or result.value["frame"]
        # pil_image = PImage.open(BytesIO(frame))
        # 
        # print("\ryou render came back with keys: [", end="")
        # print(*result.value.keys(), sep=", ", end="]")
        # 
        # frame = result.value["depthFrame"] or result.value["frame"]

        while True:
            await asyncio.sleep(1)


    app.run()
