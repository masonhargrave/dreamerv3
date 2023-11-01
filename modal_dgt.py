import modal
from modal import Image
import warnings
import os
import shutil
import dreamerv3
from dreamerv3 import embodied

logs_volume = modal.NetworkFileSystem.persisted("tensorboard-logs")
VOLUME_ROOT = "/nfs-root"

dreamer_image = Image.from_registry("masonharg/dreamerv3:latest")

# Create a modal stub
stub = modal.Stub("graphworld-training")



@stub.function(gpu="T4", network_file_systems={VOLUME_ROOT: logs_volume}, image=dreamer_image, timeout=86400)
def train_agent(N_ROOMS, SEED):

    
    # Dynamically construct the log directory path based on provided arguments
    LOGS_DIR = f"{VOLUME_ROOT}/logdir/run_{N_ROOMS}_{SEED}"

    # Ensure the directory exists
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    # See configs.yaml for all options.
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['small'])
    config = config.update({
        'logdir': LOGS_DIR,
        'run.log_every': 300,  # Seconds
        'run.steps': 1e7,
        'run.train_fill': 10000, # How much to prefill dataset
        'encoder.mlp_keys': 'vector',
        'decoder.mlp_keys': 'vector',
        'encoder.cnn_keys': '$^',
        'decoder.cnn_keys': '$^',
    })
    # config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
    ])
    import sys
    sys.path.append("/embodied")
    import graphworld
    #import graphworld
    from embodied.envs import from_gym
    env = graphworld.GraphWorld(N_ROOMS, SEED)
    env = from_gym.FromGym(env, obs_key='vector')
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(
        config.batch_length, config.replay_size, logdir / 'replay')
    args = embodied.Config(
        **config.run, logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length)
    embodied.run.train(agent, env, replay, logger, args)
    # embodied.run.eval_only(agent, env, logger, args)

    return "Training Complete!"

@stub.local_entrypoint()
def main():
    N_ROOMS = int(input("Enter number of rooms: "))
    SEED = int(input("Enter seed: "))

    print("N_ROOMS: ", N_ROOMS)
    print("SEED: ", SEED)

    result = train_agent.remote(N_ROOMS, SEED)
    print(result)
  
