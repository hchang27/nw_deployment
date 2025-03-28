# Experiments

Structure:

- first run collect trajectories
- generate views
- simulated eval

## Next Steps

- [ ]  take all four vuer scenes,
  - [ ]  create mujoco envs,
  - [ ]  label the tools / how you use them, and
  - [ ]  start adding the waypoints
- [ ]  document the tooling required to go from real to sim in MuJoCo

We don't need the height map and isaacgym if we dont' use isaacgym for rollouts. With simple terrain geometry MuJoCo suffice.

next steps:

- [ ]  either use simple polygons to roughen mujoco, or
- [ ]  backport the vuer scene definition into isaacgym


- [X]  finish all of our scenes in vuer (Ge drag this in, save as json files)

  - [ ]  take exmaple height map + way points, load in isaacgym

- setup the MuJoCo envs from our scenes (all of it, in a nice set)
  - generate the json files
  - load json files -> mjcf files
- we need to redo the trajectory collection using the unified scene definition
  - load from these and run

- [ ]  generate images using the loaded trajectories
- [ ]  Alan build the environments, make it super simple (new Isaacgym envs)
- [ ]  (or just unroll
- [ ]  write down the instructions for porting mesh into MuJoCo

What we can do with MuJoCo envs

1. we can run the experts. This is if I want to collect trajs and render masks
2. we can run the depth agent.
3. we can run the visual agent.



features

1. Ge add the ability to hydrate a scene from a json file [ not necessary ]
2.
