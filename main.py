""" The main function of rPPG deep learning pipeline."""

import argparse
import random
import time

import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/PURE_PURE_UBFC-rPPG_FreqSepPhys.yaml",
                        type=str, help="The name of the model.")
    return parser


def get_data_loader_class(dataset_name):
    """根据数据集名称返回对应的Loader类"""
    if dataset_name == "UBFC-rPPG":
        return data_loader.UBFCrPPGLoader.UBFCrPPGLoader
    elif dataset_name == "PURE":
        return data_loader.PURELoader.PURELoader
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'! "
            f"Currently supporting UBFC-rPPG and PURE.")


def train_and_test(config, data_loader_dict):

    print(f"Start Training and Testing! {config.MODEL.NAME}", end='\n\n', )

    """Trains the model."""
    if config.MODEL.NAME == "FreqSepPhys":
        model_trainer = trainer.FreqSepPhysTrainer.FreqSepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "FreqSepPhysPlusPlus":
        model_trainer = trainer.FreqSepPhysPlusPlusTrainer.FreqSepPhysPlusPlusTrainer(config, data_loader_dict)
    else:
        raise ValueError(
            f"Unsupported model '{config.MODEL.NAME}'! "
            f"Currently only FreqSepPhys and FreqSepPhysPlusPlus are supported in this minimal codebase.")
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)


def test(config, data_loader_dict):
    print(f"Start Testing! {config.MODEL.NAME}", end='\n\n', )
    """Tests the model."""
    if config.MODEL.NAME == "FreqSepPhys":
        model_trainer = trainer.FreqSepPhysTrainer.FreqSepPhysTrainer(config, data_loader_dict)
    elif config.MODEL.NAME == "FreqSepPhysPlusPlus":
        model_trainer = trainer.FreqSepPhysPlusPlusTrainer.FreqSepPhysPlusPlusTrainer(config, data_loader_dict)
    else:
        raise ValueError(
            f"Unsupported model '{config.MODEL.NAME}'! "
            f"Currently only FreqSepPhys and FreqSepPhysPlusPlus are supported in this minimal codebase.")
    model_trainer.test(data_loader_dict)


def unsupervised_method_inference(config, data_loader):
    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        if unsupervised_method == "POS":
            unsupervised_predict(config, data_loader, "POS")
        elif unsupervised_method == "CHROM":
            unsupervised_predict(config, data_loader, "CHROM")
        elif unsupervised_method == "ICA":
            unsupervised_predict(config, data_loader, "ICA")
        elif unsupervised_method == "GREEN":
            unsupervised_predict(config, data_loader, "GREEN")
        elif unsupervised_method == "LGI":
            unsupervised_predict(config, data_loader, "LGI")
        elif unsupervised_method == "PBV":
            unsupervised_predict(config, data_loader, "PBV")
        elif unsupervised_method == "OMIT":
            unsupervised_predict(config, data_loader, "OMIT")
        else:
            raise ValueError("Not supported unsupervised method!")


if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    # configurations.
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')

    data_loader_dict = dict()  # dictionary of data loaders
    if config.TOOLBOX_MODE == "train_and_test":
        # train_loader
        train_loader = get_data_loader_class(config.TRAIN.DATA.DATASET)

        # Create and initialize the train dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset paths
        if (config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH):
            train_data_loader = train_loader(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA,
                device=config.DEVICE)
            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=16,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=True,
                worker_init_fn=seed_worker,
                generator=train_generator
            )
        else:
            data_loader_dict['train'] = None

        # valid_loader
        if config.VALID.DATA.DATASET is None and config.TEST.USE_LAST_EPOCH:
            data_loader_dict['valid'] = None
        elif config.VALID.DATA.DATASET is None and not config.TEST.USE_LAST_EPOCH:
            raise ValueError("Validation dataset not specified despite USE_LAST_EPOCH set to False!")
        else:
            valid_loader = get_data_loader_class(config.VALID.DATA.DATASET)
            # Create and initialize the valid dataloader
            if (config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH):
                valid_data = valid_loader(
                    name="valid",
                    data_path=config.VALID.DATA.DATA_PATH,
                    config_data=config.VALID.DATA,
                    device=config.DEVICE)
                data_loader_dict["valid"] = DataLoader(
                    dataset=valid_data,
                    num_workers=16,
                    batch_size=config.TRAIN.BATCH_SIZE,
                    shuffle=False,
                    worker_init_fn=seed_worker,
                    generator=general_generator
                )
            else:
                data_loader_dict['valid'] = None

    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":
        # test_loader
        test_loader = get_data_loader_class(config.TEST.DATA.DATASET)

        if config.TOOLBOX_MODE == "train_and_test" and config.TEST.USE_LAST_EPOCH:
            print("Testing uses last epoch, validation dataset is not required.", end='\n\n')

        # Create and initialize the test dataloader
        if config.TEST.DATA.DATASET and config.TEST.DATA.DATA_PATH:
            test_data = test_loader(
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA,
                device=config.DEVICE)
            data_loader_dict["test"] = DataLoader(
                dataset=test_data,
                num_workers=16,
                batch_size=config.INFERENCE.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['test'] = None

    elif config.TOOLBOX_MODE == "unsupervised_method":
        # unsupervised method dataloader
        unsupervised_loader = get_data_loader_class(config.UNSUPERVISED.DATA.DATASET)

        unsupervised_data = unsupervised_loader(
            name="unsupervised",
            data_path=config.UNSUPERVISED.DATA.DATA_PATH,
            config_data=config.UNSUPERVISED.DATA,
            device=config.DEVICE)
        data_loader_dict["unsupervised"] = DataLoader(
            dataset=unsupervised_data,
            num_workers=16,
            batch_size=1,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=general_generator
        )

    else:
        raise ValueError("Unsupported toolbox_mode! Currently support train_and_test or only_test or unsupervised_method.")

    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "unsupervised_method":
        unsupervised_method_inference(config, data_loader_dict)
    else:
        print("TOOLBOX_MODE only support train_and_test or only_test !", end='\n\n')