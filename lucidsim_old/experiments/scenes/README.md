# LucidSim Domains

|  Chasing Balls | Stairs (cones) | Gaps                                                            |
| -------------- | -------------- |-----------------------------------------------------------------|
| ![ComfyUI_temp_pfflm_01247_.png](ball_scene%2Fsamples%2FComfyUI_temp_pfflm_01247_.png) | ![ComfyUI_temp_psfdt_01235_.png](cone_scene%2Fsamples%2FComfyUI_temp_psfdt_01235_.png) | ![ComfyUI_04695_.png](gap_scene%2Fsamples%2FComfyUI_04695_.png) | 

Todos:
- [ ] Close the loop of the first domain, including 
  - sampling the teacher trajectories (run different script, comparison, ask Alan to run after I set up).
  - run full eval.
  - run deployment (one week later)
- [ ] Second domain: cone following on flat ground
- [ ] Third domain: gap, cone following
- [ ] Fourth domain: stairs, cone following.

We should have a combination of cone-following vs remote controlled polices. 

Alan:
- [ ] **House Keeping**: spend some time to 
  - clean up the environments, 
  - write documentation for each, 
  - work with Ge to discuss each environment, so that Ge can suguest new terrains to experiment with.
    > Ge: I prefer to have a simple API for the terrains. and documentation.

- [ ] using chatGPT to “expand” text prompts is such a popular technique now, we should definitely experiment with it. 
  - @Alan Yu do this programmatically, ask Ran for an example. She has been working on this recently and has example scripts for agents.