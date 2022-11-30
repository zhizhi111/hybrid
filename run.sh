export HOROVOD_CACHE_CAPACITY=0

if [ -d "vqa_output_soft_hybrid_vinvl20000" ]; then
rm -rf vqa_output_soft_hybrid_vinvl20000
fi

python train_vqa.py --config config/train-vqa-base-4gpu.json \
    --output_dir vqa_output_soft_hybrid_vinvl20000