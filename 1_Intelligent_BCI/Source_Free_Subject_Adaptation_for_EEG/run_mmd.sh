model_path_source="./models/mmd/source/sub_0"
output_path_source="./outputs/mmd/source/sub_0"

model_path_g="./models/mmd/generator/k_1/sub_0"
output_path_g="./outputs/mmd/generator/k_1/sub_0"

model_path_target="./models/mmd/target/k_1/sub_0"
output_path_target="./outputs/mmd/target/k_1/sub_0"

seed=7
k=1
target_subject=0

# Source model training: source data will be unavailable after this step
CUDA_VISIBLE_DEVICES=1 python train_source.py --model_path ${model_path_source} --output_path ${output_path_source} --k ${k} --target_subject ${target_subject} --seed ${seed}

# Generator training
CUDA_VISIBLE_DEVICES=1 python train_generator.py --model_path ${model_path_g} --output_path ${output_path_g} --k ${k} --target_subject ${target_subject} --seed ${seed} --ckpt_path_source ${model_path_source} --lr '[0.01]*200'

# Target subject adaptation
CUDA_VISIBLE_DEVICES=1 python train_target.py --model_path ${model_path_target} --output_path ${output_path_target} --k ${k} --target_subject ${target_subject} --seed ${seed} --ckpt_path_g ${model_path_g} --mmd
