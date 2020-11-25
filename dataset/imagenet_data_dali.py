import os.path as osp
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, world_size, device_id, data_dir, 
                data_cfg):
        super(HybridTrainPipe, self).__init__(batch_size,
                                                data_cfg.num_threads,
                                                device_id,
                                                seed=12 + device_id)
        self.input = ops.MXNetReader(path = osp.join(data_dir, data_cfg.train_pref+".rec"), 
                                    index_path = osp.join(data_dir, data_cfg.train_pref+".idx"),
                                    random_shuffle = True, 
                                    shard_id = device_id, 
                                    num_shards = world_size)
        #let user decide which pipeline works him bets for RN version he runs
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.rrc = ops.RandomResizedCrop(device='gpu',
                            size=data_cfg.input_size[1],
                            random_area=[data_cfg.random_sized.min_scale, 1.0],
                            # interp_type=getattr(types, data_cfg.type_interp),
                            min_filter=getattr(types, data_cfg.min_filter),
                            mag_filter=getattr(types, data_cfg.mag_filter),
                            minibatch_size=data_cfg.resize_batch)
        if data_cfg.color:
            self.color = ops.ColorTwist(device="gpu")
            self.bright = ops.Uniform(range=[0.6, 1.4])
            self.cont = ops.Uniform(range=[0.6, 1.4])
            self.sat = ops.Uniform(range=[0.6, 1.4])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                        output_layout=types.NCHW,
                                        mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                        std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.rrc(images)
        if hasattr(self, 'color'):
            images = self.color(images,
                                brightness=self.bright(),
                                contrast=self.cont(),
                                saturation=self.sat())
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, world_size, device_id, data_dir, data_cfg):
        super(HybridValPipe, self).__init__(batch_size,
                                            data_cfg.num_threads,
                                            device_id,
                                            seed=12 + device_id)
        self.input = ops.MXNetReader(path = osp.join(data_dir, data_cfg.val_pref+".rec"), 
                                    index_path = osp.join(data_dir, data_cfg.val_pref+".idx"),
                                    random_shuffle = data_cfg.val_shuffle, 
                                    shard_id = device_id, 
                                    num_shards = world_size)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu",
                            resize_shorter=256,
                            # interp_type=getattr(types, data_cfg.type_interp),
                            min_filter=getattr(types, data_cfg.min_filter),
                            mag_filter=getattr(types, data_cfg.mag_filter),
                            minibatch_size=data_cfg.resize_batch)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                        output_layout=types.NCHW,
                                        crop=(data_cfg.input_size[1],)*2,
                                        mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                        std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]

class DALIClassificationIterator_(DALIClassificationIterator):
    def next_batch(self):
        try:
            data = self.__next__()
            input = data[0]["data"]
            target = data[0]["label"].squeeze().cuda().long()
        except StopIteration:
            input, target = None, None
        return input, target


def get_train_loader(batch_size, world_size, device_id, 
                    data_dir, data_cfg):
    pipe = HybridTrainPipe(batch_size=batch_size,
                           world_size=world_size,
                           device_id=device_id,
                           data_dir=data_dir,
                           data_cfg=data_cfg,)
    pipe.build()
    train_loader = DALIClassificationIterator_(pipe, reader_name="Reader", 
                fill_last_batch=False, auto_reset=True)
    return train_loader


def get_val_loader(batch_size, world_size, device_id, 
                    data_dir, data_cfg):
    pipe = HybridValPipe(batch_size=batch_size,
                        world_size=world_size,
                        device_id=device_id,
                        data_dir=data_dir,
                        data_cfg=data_cfg)
    pipe.build()
    val_loader = DALIClassificationIterator_(pipe, reader_name="Reader", 
                fill_last_batch=False, auto_reset=True)
    return val_loader

def get_data_loader(data_cfg, world_size, device_id, data_dir):
    train_loader = get_train_loader(data_cfg.batch_size,
                                world_size, device_id,
                                data_dir, data_cfg)
    val_loader = get_val_loader(data_cfg.batch_size,
                                world_size, device_id,
                                data_dir, data_cfg)
    return train_loader, val_loader