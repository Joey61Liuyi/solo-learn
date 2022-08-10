python3 main_linear.py \
    --dataset imagenet100 \
    --backbone resnet18 \
    --train_data_path /datasets/imagenet-100/train \
    --val_data_path /datasets/imagenet-100/val \
    --max_epochs 100 \
    --devices 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.3 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 512 \
    --num_workers 5 \
    --data_format dali \
    --name vicreg-imagenet100-linear-eval \
    --pretrained_feature_extractor $1 \
    --project solo-learn \
    --entity unitn-mhug \
    --wandb \
    --save_checkpoint \
    --auto_resume
