import sys
import warnings
import dreamerv3
from dreamerv3 import embodied
from treat_rl import DiseaseTreatmentEnv
from embodied.envs import from_gym
import argparse

import modal
logs_volume = modal.NetworkFileSystem.persisted("tensorboard-logs")
VOLUME_ROOT = "/nfs-root"
dreamer_image = modal.Image.from_registry("masonharg/dreamerv3:modal")
stub = modal.Stub("disease-training")

@stub.function(gpu="T4", network_file_systems={VOLUME_ROOT: logs_volume}, image=dreamer_image, timeout=86400)
def train_agent_modal():
   print("Runnning in Modal...")
   main(remote=True)

def main(remote=False):
  
  BASE_LOGS_DIR = "/logdir/disease_medium_7_3_5_1"
  if remote:
    LOGS_DIR = f"{VOLUME_ROOT}{BASE_LOGS_DIR}"
  else:
    LOGS_DIR = f"~{BASE_LOGS_DIR}"
  
  print("Running main function...")
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])

  # See configs.yaml for all options.
  config = config.update({
      'logdir': LOGS_DIR,
      'run.log_every': 30,  # Seconds
      'encoder.mlp_keys': 'symptoms',
      'decoder.mlp_keys': 'symptoms',
      'encoder.cnn_keys': '^$',
      'decoder.cnn_keys': '^$',
  })

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(),
      embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
  ])

  env = DiseaseTreatmentEnv() 
  env = from_gym.FromGym(env, obs_key='symptoms') 
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

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--local", action="store_true", help="Run locally")
  args, unknown = parser.parse_known_args()
  if args.local:
    main()
  else:
    train_agent_modal.remote()