gpu_id=0
master_port=8900
CUDA_VISIBLE_DEVICES=${gpu_id} torchrun --nproc_per_node=1 --master_port=${master_port} main_direct.py --conf_path ./config/dermamnist_resnet18.hocon