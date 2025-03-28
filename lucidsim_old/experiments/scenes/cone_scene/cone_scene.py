from vuer.schemas import Box, Cylinder, Plane, PointLight, group, Scene

scene = Scene(group(
    PointLight(color="white", intensity=10, position=[0, 2, 0]),
    PointLight(color="white", intensity=15, position=[-3, 2, 2.5]),
    Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625, 1], key="cone-1", materialType="standard",
             material=dict(emissive="#222", roughness=0.05, color="#FFA500")),
    Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625 + 0.45, 0], key="cone-2", materialType="standard",
             material=dict(emissive="#222", roughness=0.05, color="#FFA500")),
    Cylinder(args=[0.04, 0.1, 0.25, 20], position=[0, 0.625 + 0.9, -1], key="cone-3", materialType="standard",
             material=dict(emissive="#222", roughness=0.05, color="#FFA500")),
    Box(args=[3, 0.7, 1], position=[0, 0.15, 1], key="step-1", materialType="standard",
        material=dict(emissive="#222", roughness=0.01, color="#b1b1c2")),
    Box(args=[3, 0.7, 1], position=[0, 0.6, 0], key="step-2", materialType="standard",
        material=dict(emissive="#222", roughness=0.01, color="#b1b1c2")),
    Box(args=[3, 0.7, 1], position=[0, 1.1, -1], key="step-3", materialType="standard",
        material=dict(emissive="#222", roughness=0.01, color="#b1b1c2")),
    Plane(args=[100, 100], rotation=[-3.14 / 2, 0, 0], materialType="standard",
          material=dict(color="#b1b1c2", emissive="#222", roughness=0.1)),
    key='scene-group'
), up=[0, 0, 1]).serialize()

if __name__ == '__main__':
    
    # write to json
    import json

    with open("cone_scene.json", "w") as f:
        json.dump(scene, f, indent=2)
