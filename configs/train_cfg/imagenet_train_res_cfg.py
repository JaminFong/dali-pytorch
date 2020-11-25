_base_ = './imagenet_train_base_cfg.py'

model_zoo=dict(
    use_model_zoo = True,
    model_name = 'resnet18',
    cfg = None
)

train_params=dict(
    epochs=240,
)

optim=dict(
    weight_decay=4e-5,
    bn_wo_wd=False,
    if_resume=False,
    resume=dict(
        load_path='',
        load_epoch=191,
    ),
    use_warm_up=False,
    warm_up=dict(
        epoch=5,
        init_lr=1e-5,
        target_lr=0.5,
    ),
)

data=dict(
    num_threads=4,
    resize_batch=16, # 32 in default
    batch_size=128,
    color=False,
)