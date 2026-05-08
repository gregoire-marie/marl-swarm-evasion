import argparse
import logging

import ray

from src.main.python.experiment.train import (
    add_execution_args,
    add_scenario_args,
    add_training_args,
    launch_training,
    setup_training,
    teardown_training,
)
from src.main.python.utils.helpers import get_logger


logger = get_logger("train_app", level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MARL agents in an orbital pursuit-evasion scenario.")
    add_scenario_args(parser)
    add_training_args(parser)
    add_execution_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    ray.init(ignore_reinit_error=True)

    training_ctx = {}
    try:
        training_ctx = setup_training(args)
        spec = training_ctx["spec"]

        logger.info(f"Initialized training with {spec.n_interceptors} interceptors and {spec.n_targets} targets.")
        logger.info(f"Interceptor Obs Space: {training_ctx['interceptor_obs_space']}")
        logger.info(f"Target Obs Space: {training_ctx['target_obs_space']}")
        logger.info(f"Maneuver frame: {spec.maneuver_frame}")
        logger.info(f"Targets maneuvering disabled: {spec.freeze_targets}")
        logger.info(f"Results will be saved to: {training_ctx['results_dir']}")

        training_ctx = launch_training(training_ctx)

        logger.info("Training completed successfully.")
        if "last_checkpoint_path" in training_ctx:
            logger.info(f"Last checkpoint: {training_ctx['last_checkpoint_path']}")

    finally:
        teardown_training(training_ctx)


if __name__ == "__main__":
    main()
