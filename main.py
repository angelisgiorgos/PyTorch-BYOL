import os

import torch
import yaml
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms, bkground_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import resnet18
from trainer import BYOLTrainer
from utils3d import SyntheticTrainingDataset

print(torch.__version__)
torch.manual_seed(0)


def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_simclr_data_transforms(**config['data_transforms'])

    background_transforms = bkground_transforms(**config['data_transforms'])

    # train_dataset = datasets.STL10('/home/thalles/Downloads/', split='train+unlabeled', download=True,
    #                                transform=MultiViewDataInjector([data_transform, data_transform]))
    train_dataset = SyntheticTrainingDataset(npz_path=config['dataset']['path'],
                                             textures_path=config['dataset']['texture_path'],
                                             backgrounds_dir_path=config['dataset']['background_path'],
                                             transforms=MultiViewDataInjector(
                                                 [data_transform, data_transform]),
                                            background_transforms=MultiViewDataInjector(
                                                 [background_transforms, background_transforms]))

    # dataset_train = HumanImageDataset(data_path=args.data_path)
    print(len(train_dataset))
    channels = 18

    # online network
    online_network = resnet18(in_channels=channels).to(device)
    pretrained_folder = config['network']['fine_tune_from']


    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = resnet18(in_channels=channels).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])

    trainer.train(train_dataset)


if __name__ == '__main__':
    main()
