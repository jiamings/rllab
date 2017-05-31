from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument('arch', type=str)
#args = parser.parse_args()
#arch = tuple([int(a) for a in args.arch.split('-')])

#print(arch)

def run_task(*_):
    env = normalize(MountainCarEnv())

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)#arch
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=4000,
        max_path_length=200,
        n_itr=100,
        discount=0.99,
        step_size=0.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        #plot=True
    )
    algo.train()


run_experiment_lite(
    run_task,
    #exp_prefix='mountain_car',
    #exp_name=args.arch,
    # Number of parallel workers for sampling
    n_parallel=2,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    #plot=True
)
