import asyncio

from vuer import Vuer
from vuer.schemas import Box, Plane, Scene, group, Cylinder

scene = Scene(group(
    Box(args=[1, 3, 1], position=[-0.5, 0, 0.5], rotation=[0, 0, -1.57], key="hurdle1", material=dict(color="blue")),
    Box(args=[1, 1, 1], position=[2.0, 0, 0.5], rotation=[0, 0, -1.57], key="hurdle2",
        material=dict(color="blue")),
    Box(args=[1, 1, 1], position=[3.5, 0, 0.5], rotation=[0, 0, -1.57], key="hurdle3",
        material=dict(color="blue")),
    Plane(args=[100, 100], rotation=[0, 0, 0], material=dict(color="red")),
    Cylinder(args=[0.5], position=[2, 0, 0], key="cone-1", materialType="standard"),
    Cylinder(args=[0.5], position=[4, 0, 0], key="cone-2", materialType="standard"),
    Cylinder(args=[0.5], position=[6, 0, 0], key="cone-3", materialType="standard"),
    key='scene-group',
),
    up=[0, 0, 1])

if __name__ == '__main__':
    app = Vuer()


    @app.spawn
    async def main(session):
        session.set @ scene

        while True:
            await asyncio.sleep(1)


    app.run()
