"""
RTI (Rollout-Time Intervention) async training entry point.

Slime's own ``train_async.py`` calls ``parse_args()`` without a custom-argument
provider, so the RTI arguments (``--inspector-*``, ``--ref-*``, ...) must be
registered by a dedicated entry.  This module mirrors slime's async loop and
adds:

1. ``parse_args(add_custom_arguments=add_rti_arguments)`` so the RTI CLI flags
   in ``qqr.args`` are accepted.
2. A pre-train evaluation gated by ``--skip-eval-before-train`` (same behaviour
   as slime's sync ``train.py``).

Launch (see ``scripts/<task>/run-*.sh``)::

    cd ${SLIME_PATH}
    python3 -m qqr.off_policy.train_async <slime train args...>
"""

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking
from slime.utils.misc import should_run_periodic_action

from qqr.args import add_rti_arguments


def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if not args.critic_train_only:
        actor_model.update_weights()
        if args.check_weight_update_equal:
            ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.eval_interval is not None and not getattr(args, "skip_eval_before_train", False):
        if args.start_rollout_id == 0:
            ray.get(rollout_manager.eval.remote(0))
        else:
            resume_eval_id = args.start_rollout_id - 1
            print(f"Resuming eval for last unevaluated checkpoint iter_{resume_eval_id:07d}")
            ray.get(rollout_manager.eval.remote(resume_eval_id))

    rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if rollout_data_next_future is not None:
            rollout_data_curr_ref = ray.get(rollout_data_next_future)

        if rollout_id + 1 < args.num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_curr_ref)
            if rollout_id >= args.num_critic_only_steps and not args.critic_train_only:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            if not args.critic_train_only:
                actor_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.use_critic:
                critic_model.save_model(
                    rollout_id,
                    force_sync=rollout_id == args.num_rollout - 1,
                )
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        if (rollout_id + 1) % args.update_weights_interval == 0:
            rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
            rollout_data_next_future = None
            if not args.critic_train_only:
                actor_model.update_weights()

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args(add_custom_arguments=add_rti_arguments)
    train(args)
