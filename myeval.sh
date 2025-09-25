export HF_ENDPOINT=https://hf-mirror.com
export HEADLESS=1
export HF_TOKEN=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
export WANDB_API_KEY=f02cb01552baf522a9cf61ea54648decbdb3c7e9
export HF_HOME=cache
export CUDA_VISIBLE_DEVICES=1
# export DISPLAY=:0
# export XDG_RUNTIME_DIR=/usr/lib
# export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# python experiments/robot/libero/run_libero_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint logs/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-object_old-vq7-mixed90-vggt-cross-freeze_v-new_cat-camera-head_rcrop--image_aug/checkpoints/latest-checkpoint.pt \
#   --task_suite_name libero_object \
#   --center_crop True

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint logs/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-object_old-vggt-cross-freeze_v-new_cat-camera-head_rcrop--image_aug/checkpoints/latest-checkpoint.pt \
  --task_suite_name libero_object \
  --center_crop True

# python experiments/robot/libero/run_libero_eval_old.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
#   --task_suite_name libero_spatial \
#   --center_crop True


# python experiments/robot/simpler/run_simpler_eval.py \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b \
#   --task_suite_name simpler_widowx
# #  --center_crop True

# python experiments/robot/libero/regenerate_libero_dataset_v2_dp.py \
#   --libero_target_dir experiments/robot/libero/datasets/libero_90_no_noops_dp \
#   --libero_raw_data_dir experiments/robot/libero/datasets/libero_90 \
#   --libero_task_suite libero_90
