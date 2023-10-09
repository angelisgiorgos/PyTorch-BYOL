import os

import torch
import yaml
from torchvision import datasets
<<<<<<< HEAD
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms, bkground_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import resnet18
from trainer import BYOLTrainer
from utils3d import SyntheticTrainingDataset
=======
from data.multi_view_data_injector import MultiViewDataInjector, Multi3DData
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import resnet18
from trainer import BYOLTrainer
from utils3d import SyntheticTrainingDataset, TexturedIUVRenderer, SMPL
from data.transforms import get_simclr_data_3dtransforms
>>>>>>> 3c27d33b631e7063f55844bab34aa02b94b5fea6

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
<<<<<<< HEAD
                                             textures_path=config['dataset']['texture_path'],
                                             backgrounds_dir_path=config['dataset']['background_path'],
                                             transforms=MultiViewDataInjector(
                                                 [data_transform, data_transform]),
                                            background_transforms=MultiViewDataInjector(
                                                 [background_transforms, background_transforms]))

=======
                                             transforms=Multi3DData(
                                                 [data_transform, data_transform]))
>>>>>>> 3c27d33b631e7063f55844bab34aa02b94b5fea6
    # dataset_train = HumanImageDataset(data_path=args.data_path)
    print(len(train_dataset))
    channels = 18

    # online network
    online_network = resnet18(in_channels=channels).to(device)
    pretrained_folder = config['network']['fine_tune_from']

<<<<<<< HEAD
=======
    # load pre-trained model if defined
    # if pretrained_folder:
    #     try:
    #         checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')
    #
    #         # load pre-trained parameters
    #         load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
    #                                  map_location=torch.device(torch.device(device)))
    #
    #         online_network.load_state_dict(load_params['online_network_state_dict'])
    #
    #     except FileNotFoundError:
    #         print("Pre-trained weights not found. Training from scratch.")

>>>>>>> 3c27d33b631e7063f55844bab34aa02b94b5fea6
    # predictor network
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
