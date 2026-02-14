from tensorboardX import SummaryWriter
import os

def log_tensorboard_layout(logdir: str):
    """
    Defines and logs a custom layout for TensorBoard to group orbital metrics.
    
    This creates a 'Custom Scalars' tab in TensorBoard with organized charts
    for success rates, collisions, and other domain-specific metrics.
    
    Args:
        logdir (str): The directory where TensorBoard event files are stored.
                     Should be the root directory of the Ray results.
    """
    layout = {
        "Orbital Metrics": {
            "Success and Failures": ["Multiline", [
                "ray/tune/env_runners/intercept_success_rate",
                "ray/tune/env_runners/out_of_fuel_rate"
            ]],
            "Collisions": ["Multiline", [
                "ray/tune/env_runners/interceptors_collision_rate",
                "ray/tune/env_runners/targets_collision_rate"
            ]],
            "Episode Metrics": ["Multiline", [
                "ray/tune/env_runners/episode_steps",
                "ray/tune/env_runners/episode_len_mean"
            ]],
            "Overall Training Performance": ["Multiline", [
                "ray/tune/env_runners/episode_return_mean"
            ]],
            "Interceptor Agents": ["Multiline", [
                "ray/tune/learners/interceptor_policy/policy_loss",
                "ray/tune/learners/interceptor_policy/mean_kl_loss",
                "ray/tune/learners/interceptor_policy/entropy",
                "ray/tune/learners/interceptor_policy/curr_kl_coeff",
                "ray/tune/learners/interceptor_policy/vf_loss_unclipped",
                "ray/tune/learners/interceptor_policy/total_loss",
            ]],
            "Target Agents": ["Multiline", [
                "ray/tune/learners/target_policy/policy_loss",
                "ray/tune/learners/target_policy/mean_kl_loss",
                "ray/tune/learners/target_policy/entropy",
                "ray/tune/learners/target_policy/curr_kl_coeff",
                "ray/tune/learners/target_policy/vf_loss_unclipped",
                "ray/tune/learners/target_policy/total_loss",
            ]]
        },
    }

    # Ensure logdir exists
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)

    writer = SummaryWriter(log_dir=logdir)
    writer.add_custom_scalars(layout)
    writer.close()
