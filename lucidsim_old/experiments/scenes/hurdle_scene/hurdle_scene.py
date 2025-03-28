from vuer.schemas import Box, Cylinder, Plane, PointLight, group, Scene

scene = Scene(group(
    Box(args=[15, 15, 15], position=[0, 0.5, 0], materialType="depth", key="room", material=dict(side=1)),
    Box(args=[15, 1, 1], position=[0, 0.5, 0], materialType="depth", key="hurdle"),
    Plane(args=[100, 100], rotation=[-3.14 / 2, 0, 0], materialType="depth"),
    key='scene-group'
), up=[0, 0, 1]).serialize()

# write to json
import json

with open("hurdle_scene.json", "w") as f:
    json.dump(scene, f, indent=2)
