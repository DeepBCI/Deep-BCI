from torch.utils.data import DataLoader

from data_loader.dataset.bcic4_2a import BCIC4_2A


class DataGenerator:
    def __init__(self, args):
        print("[Load Data]")
        if args.mode == 'train':
            self.train_loader = self.__data_loader(args, 'train')
            self.val_loader = self.__data_loader(args, 'test')
            print(f"train size: {self.train_loader.dataset.X.shape}")
            print(f"val size: {self.val_loader.dataset.X.shape}")
            mini_batch_shape = list(self.train_loader.dataset.X.shape)
            mini_batch_shape[0] = args.batch_size
            args.cfg.input_shape = mini_batch_shape
            print("")
        else:
            self.test_loader = self.__data_loader(args, 'test')
            print(f"test size: {self.test_loader.dataset.X.shape}")
            print("")

    def __data_loader(self, args, phase):
        return DataLoader(BCIC4_2A(args, phase),
                          batch_size=args.batch_size,
                          shuffle=True if phase == 'train' else False,
                          drop_last=True if phase == 'train' else False)
