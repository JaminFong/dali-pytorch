_base_ = './imagenet_train_base_cfg.py'

model_zoo=dict(
    use_model_zoo = True,
    model_name = 'ghostnet',
)

train_params=dict(
    epochs=400,
)

optim=dict(
    weight_decay=4e-5,
    bn_wo_wd=True,
    if_resume=False,
    resume=dict(
        load_path='',
        load_epoch=191,
    ),
    use_warm_up=True,
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
    color=True,
)