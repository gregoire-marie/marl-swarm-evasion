import argparse
import logging
import os

import ray

from src.main.python.experiment.curriculum import CurriculumTrainingConfig, CurriculumTrainingParameters
from src.main.python.experiment.train import (
    launch_training,
    setup_training,
    teardown_training,
)
from src.main.python.utils.callbacks import CurriculumCallbacks
from src.main.python.utils.helpers import get_logger


logger = get_logger("curriculum_train_app", level=logging.INFO)
TORCH_THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "TORCH_NUM_THREADS")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MARL agents with callback-based curriculum learning.")
    parser.add_argument(
        "--curriculum-config",
        type=str,
        required=True,
        help="Path to the required curriculum JSON configuration.",
    )
    return parser.parse_args()


def configure_torch_threads(torch_num_threads: int) -> None:
    for env_var in TORCH_THREAD_ENV_VARS:
        os.environ[env_var] = str(torch_num_threads)

    import torch

    torch.set_num_threads(torch_num_threads)


def build_ray_init_kwargs(parameters: CurriculumTrainingParameters):
    kwargs = {"ignore_reinit_error": True}
    if parameters.ray_num_cpus is not None:
        kwargs["num_cpus"] = parameters.ray_num_cpus
    if parameters.torch_num_threads is not None:
        thread_count = str(parameters.torch_num_threads)
        kwargs["runtime_env"] = {
            "env_vars": {env_var: thread_count for env_var in TORCH_THREAD_ENV_VARS}
        }
    return kwargs


def main():
    cli_args = parse_args()
    run_config = CurriculumTrainingConfig.from_json_file(cli_args.curriculum_config)
    curriculum_config = run_config.curriculum
    training_args = run_config.training.to_namespace()

    if run_config.training.torch_num_threads is not None:
        configure_torch_threads(run_config.training.torch_num_threads)

    ray.init(**build_ray_init_kwargs(run_config.training))

    training_ctx = {}
    try:
        training_ctx = setup_training(
            training_args,
            curriculum_config=curriculum_config,
            curriculum_parameters=run_config.to_dict(),
            callbacks_cls=CurriculumCallbacks,
        )
        spec = training_ctx["spec"]

        logger.info(
            f"Initialized curriculum training with fixed capacity "
            f"{curriculum_config.N_max} interceptors and {curriculum_config.M_max} targets."
        )
        logger.info(f"Curriculum stages: {[stage.stage_id for stage in curriculum_config.stages]}")
        if run_config.training.ray_num_cpus is not None:
            logger.info(f"Ray CPU budget: {run_config.training.ray_num_cpus}")
        if run_config.training.torch_num_threads is not None:
            logger.info(f"PyTorch thread budget: {run_config.training.torch_num_threads}")
        logger.info(f"Interceptor Obs Space: {training_ctx['interceptor_obs_space']}")
        logger.info(f"Target Obs Space: {training_ctx['target_obs_space']}")
        logger.info(f"Action-space max delta-v: {spec.max_delta_v_mps}")
        logger.info(f"Results will be saved to: {training_ctx['results_dir']}")

        training_ctx = launch_training(training_ctx)

        logger.info("Curriculum training completed successfully.")
        if "last_checkpoint_path" in training_ctx:
            logger.info(f"Last checkpoint: {training_ctx['last_checkpoint_path']}")

    finally:
        teardown_training(training_ctx)


if __name__ == "__main__":
    main()
