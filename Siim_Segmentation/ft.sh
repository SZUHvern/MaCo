bash tools/dist_train.sh \
    configs/MaCo/upernet_MaCo-base_fp16_8x2_512x512_160k_siim_1per.py 4 \
    --work-dir ./output/ --seed 0  --deterministic \
    --options model.backbone.pretrained=/mnt/disk2/hwj/MaCo-pytorch-main/output_dir/model/example-model.pth
