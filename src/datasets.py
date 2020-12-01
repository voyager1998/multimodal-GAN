from PIL import Image
import numpy as np
import torch
import glob
import pdb

# Dataloader for Edge2Shoe datasets=


class Edge2Shoe(torch.utils.data.Dataset):
    def __init__(self, img_dir):
        image_list = []
        for img_file in glob.glob(str(img_dir) + '*'):
            image_list.append(img_file)
        self.image_list = image_list

    def __getitem__(self, index):
        # resize images from 256 to 128 for faster training purposes
        image = Image.open(self.image_list[index]).resize((256, 128), resample=Image.BILINEAR)
        image = np.asarray(image).transpose(2, 0, 1)
        image_tensor = torch.from_numpy(image).float()
        edge_tensor = image_tensor[:, :, :128]
        rgb_tensor = image_tensor[:, :, 128:]
        return edge_tensor, rgb_tensor

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    img_dir = 'data/edges2shoes/train/'
    dataset = Edge2Shoe(img_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    for idx, data in enumerate(loader):
        edge_tensor, rgb_tensor = data
        print(idx, edge_tensor.shape, rgb_tensor.shape)
