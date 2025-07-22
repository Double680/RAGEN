set -e

USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.28 actor_rollout_ref.rollout.rollout_filter_ratio=1"
MODEL_PATH="model_path=/root/.cache/qwen3-4b"

python train.py --config-name hybtqa system.CUDA_VISIBLE_DEVICES=\"0,1,2,3,4,5,6,7\" trainer.n_gpus_per_node=8 trainer.experiment_name=hybtqa-grpo actor_rollout_ref.rollout.tensor_model_parallel_size=8 $USE_GRPO $USE_BASE $MODEL_PATH