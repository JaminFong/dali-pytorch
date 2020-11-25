model_zoo=dict(
    use_model_zoo = False,
    model_name = '',
    cfg=dict(
        init_mode='he_fout',
        bn_momentum=0.1,
        bn_eps=1e-3
    )
)

net_type='mbv2' # mbv2 / res
net_config=""""""

train_params=dict(
    epochs=240,
    use_seed=True,
    seed=0
)

optim=dict(
    init_lr=0.5,
    min_lr=1e-5,
    lr_schedule='cosine',  # cosine poly
    momentum=0.9,
    weight_decay=4e-5,
    bn_wo_wd=False,
    label_smooth=True,
    smooth_alpha=0.1,
    use_grad_clip=False,
    grad_clip=10,
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
    use_multi_stage=False,
    multi_stage=dict(
        stage_epochs=330
    ),
    cosine=dict(
        use_restart=False,
        restart=dict(
            lr_period=[10, 20, 40, 80, 160, 320],
            lr_step=[0, 10, 30, 70, 150, 310],
        )
    )
)

data=dict(
    use_dali=True,
    num_threads=4,
    resize_batch=16, # 32 in default
    batch_size=128,
    dataset='imagenet', #imagenet
    train_pref='train_orig',
    val_pref='val_orig',
    num_examples=1281167,
    input_size=(3,224,224),
    # type_interp='INTERP_TRIANGULAR' # INTERP_TRIANGULAR INTERP_LINEAR
    min_filter='INTERP_TRIANGULAR', # INTERP_TRIANGULAR INTERP_LINEAR
    mag_filter='INTERP_LANCZOS3', # INTERP_TRIANGULAR INTERP_LINEAR
    random_sized=dict(
        min_scale=0.08
    ),
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    color=False,
    val_shuffle=False
)