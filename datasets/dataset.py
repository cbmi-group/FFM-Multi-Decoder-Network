from torch.utils.data import DataLoader

from datasets.ER_dataset import ER_Dataset
from datasets.MITO_dataset import MITO_Dataset
from datasets.ROSE_dataset import ROSE_Dataset
from datasets.STARE_dataset import STARE_Dataset
from datasets.ROAD_dataset import ROAD_Dataset
from datasets.NUCLEUS_dataset import NUCLEUS_Dataset

DATASETs = {'MITO': MITO_Dataset, 'ER': ER_Dataset, 'ROSE': ROSE_Dataset,
            'SATRE': STARE_Dataset, 'ROAD': ROAD_Dataset, 'NUCLEUS': NUCLEUS_Dataset}


def build_data_loader(data_name, data_list, batch_size, dataset_type, is_train=True, fractal_dir='', weight_dir='',
                      edge_dir='', skeleton_dir='', decoder_type='',
                      log_file='', epoch=0, update_d=0):
    train_data = DATASETs[data_name](txt=data_list, dataset_type=dataset_type, train=is_train, fractal_dir=fractal_dir,
                                     weight_dir=weight_dir, edge_dir=edge_dir, skeleton_dir=skeleton_dir,
                                     decoder_type=decoder_type, log_file=log_file, epoch=epoch, update_d=update_d)

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=is_train, num_workers=8)

    return data_loader
