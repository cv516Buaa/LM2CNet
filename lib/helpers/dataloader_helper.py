import numpy as np
from torch.utils.data import DataLoader
from lib.datasets.Mono3DRefer_nuScenes.Mono3DRefer_nuScenes_dataset import Mono3DRefer_nuScenes_Dataset


# init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataloader(cfg, test_flag, workers=0):
    # perpare dataset
    if cfg['type'] == 'LM2CNet':
        train_set = Mono3DRefer_nuScenes_Dataset(split=cfg['train_split'], cfg=cfg)
        test_set = Mono3DRefer_nuScenes_Dataset(split=cfg['test_split'], cfg=cfg)
        print(f'train instance: {len(train_set)}, test instance: {len(test_set)}')
    else:
        raise NotImplementedError("%s dataset is not supported" % cfg['type'])
    if test_flag:
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=workers,
                                 worker_init_fn=my_worker_init_fn,
                                 shuffle=False,
                                 pin_memory=False,
                                 drop_last=False)
        return test_loader
    else:
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=cfg['batch_size'],
                                  num_workers=workers,
                                  worker_init_fn=my_worker_init_fn,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=False)
        test_loader = DataLoader(dataset=test_set,
                                 batch_size=cfg['batch_size'],
                                 num_workers=workers,
                                 worker_init_fn=my_worker_init_fn,
                                 shuffle=False,
                                 pin_memory=False,
                                 drop_last=False)
        return train_loader, test_loader

