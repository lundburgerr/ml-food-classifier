import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torchvision

from tqdm import tqdm
import os
import json
from PIL import Image

PATH_TO_DATASET_ = '/opt/project/datasets/food-101'
MODEL_PATH_ = 'model_food_101.pth'


class CustomDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.classes = self.load_classes()
        self.data = self.load_data()
        self.index_mapping = self.create_index_mapping()

    def load_classes(self):
        classes_file = os.path.join(self.root_dir, 'meta', 'classes.txt')
        with open(classes_file, 'r') as f:
            classes = f.read().splitlines()
        return classes

    def load_data(self):
        data_file = os.path.join(self.root_dir, 'meta', f'{self.mode}.json')
        with open(data_file, 'r') as f:
            data = json.load(f)
        return data

    def create_index_mapping(self):
        index_mapping = []
        for class_name, images in self.data.items():
            for image in images:
                img_path = os.path.join(self.root_dir, 'images', f'{image}.jpg')
                label = self.classes.index(class_name)
                index_mapping.append((img_path, label))
        return index_mapping

    @staticmethod
    def gen_transforms():
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to a larger size
            transforms.CenterCrop((224, 224)),  # Center crop to [224, 224]
            transforms.ToTensor(),  # Convert to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        return transform

    def __len__(self):
        return sum(len(images) for images in self.data.values())

    def __getitem__(self, idx):
        img_path, label = self.index_mapping[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def main():
    model = torchvision.models.efficientnet_b0(pretrained=True)
    transform = CustomDataset.gen_transforms()
    train_dataset = CustomDataset(root_dir=PATH_TO_DATASET_, mode='train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda')
    model.to(device)

    log_dir = 'logs/'
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    num_epochs = 10
    checkpoint_dir = 'checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Initialize the progress bar for the current epoch
        progress_bar = tqdm(total=len(train_loader), desc=f'Epoch [{epoch + 1}/{num_epochs}]', position=0, leave=True)

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Update the progress bar
            progress_bar.update(1)
            progress_bar.set_postfix({'Loss': running_loss / len(train_loader.dataset)})

        # Close the progress bar for the current epoch
        progress_bar.close()

        # Save loss to TensorBoard
        average_loss = running_loss / len(train_loader.dataset)
        writer.add_scalar('Loss/train', average_loss, epoch + 1)

        # Save a checkpoint at the end of each epoch
        checkpoint_name = f'checkpoint_epoch_{epoch + 1}.pth'
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path)

    # Save the model at the end of training
    torch.save(model.state_dict(), MODEL_PATH_)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
