from vuer.schemas import Box, Plane, PointLight, Scene, group

scene = Scene(group(
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
), up=[0, 0, 1]).serialize()

# write to json
import json

with open("gap_scene.json", "w") as f:
    json.dump(scene, f, indent=2)
