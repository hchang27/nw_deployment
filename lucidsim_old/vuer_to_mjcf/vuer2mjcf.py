from typing import Literal

import numpy as np
from bs4 import BeautifulSoup
from params_proto import ParamsProto, Proto

from lucidsim_old.vuer_to_mjcf.utils.primitives import VUER_PRIMITIVES


class Vuer2MJCF(ParamsProto):
    output_name = "hurdle"
    serialized = Proto(help="the serialized vuer scene object")
    write = Proto(default=False, help="write the output to a file")

    angle_mode: Literal["degree", "radian"] = "radian"

    def parse(self, current_components, node, soup):
        """
        Take in the current components and node, and returns the children
        """
        if "children" not in current_components or len(current_components["children"]) == 0:
            return

        for child in current_components["children"]:
            current = soup.new_tag('body')
            current["name"] = child["key"]
            if "up" in child:
                current["zaxis"] = " ".join([str(x) for x in child["up"]])
            if "rotation" in child:
                angles = child["rotation"]
                if self.angle_mode == "degree":
                    angles = np.rad2deg(angles)
                current["euler"] = " ".join([str(x) for x in angles])
            if "position" in child:
                current["pos"] = " ".join([str(x) for x in child["position"]])
            if "tag" in child:
                geom_type = child["tag"].lower()
                if geom_type in VUER_PRIMITIVES:
                    mj_args = VUER_PRIMITIVES[geom_type](child["args"])
                    geom_tag = soup.new_tag('geom', type=geom_type, size=mj_args)
                    current.append(geom_tag)
                if geom_type == "cylinder":
                    # temporary: used as waypoint
                    current["mocap"] = "true"

            node.append(current)
            self.parse(child, current, soup)

    def main(self):
        soup = BeautifulSoup()
        root = soup.new_tag('mujoco', model=self.output_name)
        soup.append(root)

        world = soup.new_tag('worldbody')
        root.append(world)

        self.parse(self.serialized, world, soup)

        print(soup.prettify())

        if self.write:
            with open(f"{self.output_name}.xml", "w") as f:
                f.write(soup.prettify())


if __name__ == "__main__":
    # from lucidsim.experiments.scenes.cone_scene.cone_scene import scene
    # Vuer2MJCF(output_name="examples/cone", serialized=scene, write=True).main()

    # from examples.stairs import scene

    # Vuer2MJCF(output_name="examples/stairs", serialized=scene.serialize(), write=True).main()
    # 
    from examples.gap import scene

    # 
    Vuer2MJCF(output_name="examples/gap", serialized=scene.serialize(), write=True).main()
