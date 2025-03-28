from isaacgym import gymapi, gymutil

assert gymapi
gym = gymapi.acquire_gym()

sim_device_id = 1
graphics_device_id = 1

sim_params = {'sim': {'dt': 0.005,
                      'gravity': [0.0, 0.0, -9.81],
                      'physx': {'bounce_threshold_velocity': 0.5,
                                'contact_collection': 2,
                                'contact_offset': 0.01,
                                'default_buffer_size_multiplier': 5,
                                'max_depenetration_velocity': 1.0,
                                'max_gpu_contact_pairs': 8388608,
                                'num_position_iterations': 4,
                                'num_threads': 10,
                                'num_velocity_iterations': 0,
                                'rest_offset': 0.0,
                                'solver_type': 1},
                      'substeps': 1,
                      'up_axis': 1}}

params = gymapi.SimParams()
params.physx.use_gpu = True
params.physx.num_subscenes = 0
params.use_gpu_pipeline = True
gymutil.parse_sim_config(sim_params["sim"], params)

physics_engine = gymapi.SIM_PHYSX

sim = gym.create_sim(sim_device_id, graphics_device_id, physics_engine,
                     params)

while True:
    pass
