bash tools/dist_train.sh \
    configs/MaCo/upernet_MaCo-base_fp16_8x2_512x512_160k_siim_10per.py 4 \
    --work-dir ./output/ --seed 0  --deterministic \
    --options model.backbone.pretrained=/path/to/MaCo/checkpoint-50.pth