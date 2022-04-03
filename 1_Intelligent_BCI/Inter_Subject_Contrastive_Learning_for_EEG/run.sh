model_path="./models/inter-subject_contrastive/k_1/sub_0"
output_path="./outputs/inter-subject_contrastive/k_1/sub_0"

seed=7
k=1
target_subject=0

CUDA_VISIBLE_DEVICES=0 python train.py --model_path ${model_path} --output_path ${output_path} --k ${k} --target_subject ${target_subject} --seed ${seed} --contrastive
