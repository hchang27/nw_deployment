import asyncio

from vuer import Vuer
from vuer.schemas import Box, Cylinder, Plane, Scene, group

step_height = 0.2
scene = Scene(group(group(
    Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625, 1], key="cone-1", materialType="standard",
             material=dict(emissive="#222", roughness=0.05, color="#FFA500")),
    Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625 + step_height, 0], key="cone-2", materialType="standard",
             material=dict(emissive="#222", roughness=0.05, color="#FFA500")),
    Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625 + 2 * step_height, -1], key="cone-3",
             materialType="standard",
             material=dict(emissive="#222", roughness=0.05, color="#FFA500")),
    Box(args=[15, 0.7, 1], position=[0, step_height, 1], key="step-1", materialType="basic",
        material=dict(color="white")),
    Box(args=[15, 0.7, 1], position=[0, 2 * step_height, 0], key="step-2", materialType="basic",
        material=dict(color="white")),
    Box(args=[15, 0.7, 1], position=[0, 3 * step_height, -1], key="step-3", materialType="basic",
        material=dict(color="white")),
    key='scene-group',
    rotation=[1.57, 0, 0]), rotation=[0, 0, -1.57], position=[2.0, 0, -step_height]),
    Plane(args=[100, 100], materialType="basic", material=dict(color="black")),
    up=[0, 0, 1])

if __name__ == '__main__':
    app = Vuer()


    @app.spawn
    async def main(session):
        session.set @ scene

        while True:
            await asyncio.sleep(1)


    app.run()
