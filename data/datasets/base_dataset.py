from torch.utils.data import Dataset
import numpy as np
import torch

class TrainDataset(Dataset):
    def __init__(self, df, transform=None, opt=None, is_train=True):
        self.df = df
        self.file_names = df['file_path'].values
        self.is_train = is_train
        if self.is_train:
            self.labels = df[opt.target_col].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = np.load(file_path) # (6, 273, 256)
        # import pdb
        # pdb.set_trace()
        image = image.astype(np.float32)
        image = np.vstack(image).transpose((1, 0)) # (256, 1638)
        if self.transform:
            image = self.transform(image=image)['image'] # [1, 224, 224]
        else:
            image = image[np.newaxis, :, :]
            image = torch.from_numpy(image).float()

        batch = {}
        batch['input'] = image
        if self.is_train:
            label = torch.tensor(self.labels[idx]).float()
            batch['labels'] = label
        return batch