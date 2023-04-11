model_path_source="./models/inter-subject_contrastive/source/sub_0"
output_path_source="./outputs/inter-subject_contrastive/source/sub_0"

model_path_g="./models/inter-subject_contrastive/generator/k_1/sub_0"
output_path_g="./outputs/inter-subject_contrastive/generator/k_1/sub_0"

model_path_target="./models/inter-subject_contrastive/target/k_1/sub_0"
output_path_target="./outputs/inter-subject_contrastive/target/k_1/sub_0"

seed=7
k=1
target_subject=0

# Source model training: source data will be unavailable after this step
CUDA_VISIBLE_DEVICES=0 python train_source.py --model_path ${model_path_source} --output_path ${output_path_source} --k ${k} --target_subject ${target_subject} --seed ${seed}

# Generator training
CUDA_VISIBLE_DEVICES=0 python train_generator.py --model_path ${model_path_g} --output_path ${output_path_g} --k ${k} --target_subject ${target_subject} --seed ${seed} --ckpt_path_source ${model_path_source} --lr '[0.01]*200'

# Target subject adaptation
CUDA_VISIBLE_DEVICES=0 python train_target.py --model_path ${model_path_target} --output_path ${output_path_target} --k ${k} --target_subject ${target_subject} --seed ${seed} --ckpt_path_g ${model_path_g} --contrastive
