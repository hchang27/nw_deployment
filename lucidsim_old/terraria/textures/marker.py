from params_proto import PrefixProto
from vuer import Vuer
from vuer.schemas import Cylinder, DefaultScene, group


class Marker(PrefixProto):
    key: str = "cone"

    def __call__(self, cone_material=None):
        if cone_material is None:
            cone_material = dict()

        return group(
            Cylinder(args=[0.04, 0.1, 0.25, 20], rotation=[1.57, 0, 0], position=[0, 0.625, 1],
                     **cone_material),
            Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625 + 0.45, 0],
                     **cone_material),
            Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625 + 0.9, -1],
                     **cone_material), key=self.key
        )


if __name__ == '__main__':
    import asyncio

    app = Vuer()

    marker_1 = Marker()


    @app.spawn
    async def main(sess):
        sess.set @ DefaultScene(marker_1())

        while True:
            await asyncio.sleep(1.0)


    app.run()
