_base_ = './imagenet_train_base_cfg.py'

model_zoo=dict(
    use_model_zoo = False,
    model_name = 'mobilenet_v2',
    cfg = None
)

net_type='mbv2' # mbv2 / res
net_config="""[[32, 16], 'mbconv_k3_t1', [], 0, 1]|
[[16, 24], 'mbconv_k3_t6', ['mbconv_k3_t6'], 1, 2]|
[[24, 32], 'mbconv_k3_t6', ['mbconv_k3_t6', 'mbconv_k3_t6'], 2, 2]|
[[32, 64], 'mbconv_k3_t6', ['mbconv_k3_t6', 'mbconv_k3_t6', 'mbconv_k3_t6'], 3, 2]|
[[64, 96], 'mbconv_k3_t6', ['mbconv_k3_t6', 'mbconv_k3_t6'], 2, 1]|
[[96, 160], 'mbconv_k3_t6', ['mbconv_k3_t6', 'mbconv_k3_t6'], 2, 2]|
[[160, 320], 'mbconv_k3_t6', [], 3, 1]|
[[320, 1280], 'conv1_1']"""

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
    use_se=False
)

data=dict(
    num_threads=4,
    resize_batch=16, # 32 in default
    batch_size=128,
    color=False,
)