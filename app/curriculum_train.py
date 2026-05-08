import argparse
import logging

import ray

from src.main.python.experiment.curriculum import CurriculumConfig
from src.main.python.experiment.train import (
    add_execution_args,
    add_training_args,
    launch_training,
    setup_training,
    teardown_training,
)
from src.main.python.utils.callbacks import CurriculumCallbacks
from src.main.python.utils.helpers import get_logger


logger = get_logger("curriculum_train_app", level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Train MARL agents with callback-based curriculum learning.")
    parser.add_argument(
        "--curriculum-config",
        type=str,
        required=True,
        help="Path to the required curriculum JSON configuration.",
    )
    add_training_args(parser, seed_default=None)
    add_execution_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    curriculum_config = CurriculumConfig.from_json_file(args.curriculum_config)

    ray.init(ignore_reinit_error=True)

    training_ctx = {}
    try:
        training_ctx = setup_training(
            args,
            curriculum_config=curriculum_config,
            callbacks_cls=CurriculumCallbacks,
        )
        spec = training_ctx["spec"]

        logger.info(
            f"Initialized curriculum training with fixed capacity "
            f"{curriculum_config.N_max} interceptors and {curriculum_config.M_max} targets."
        )
        logger.info(f"Curriculum stages: {[stage.stage_id for stage in curriculum_config.stages]}")
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
