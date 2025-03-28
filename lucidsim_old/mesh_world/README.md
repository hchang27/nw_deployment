# Mesh World

## Organization

This module is organized as follows:

```
├── labeler.py   # Label scenes with start and goals.
├── traj_gen.py  # Unroll trajectories from start to goal.
├── ego_view.py  # Collect Egocentric RGB and Depth Images
└── imagen.py    # Generate Images from ego views. [ now in the imagen repository ]
```

The data is organized as below:

```
datasets/lucidsim
└── scenes
    └── mit_stairs
        └── stairs_0001_v1
            ├── assets
            │   ├── point_cloud.ply
            │   ├── textured.mtl
            │   ├── textured.obj
            │   └── textured_0_UiKfOKzt.jpg
            ├── ego_views
            ├── labels.yml
            ├── lucid_dreams
            ├── prompts.yml
            └── trajectories
```

~~~~

~~~~
