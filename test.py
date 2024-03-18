from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

from train import CustomDataset, PATH_TO_DATASET_, MODEL_PATH_


def main():
    transform = CustomDataset.gen_transforms()
    test_dataset = CustomDataset(root_dir=PATH_TO_DATASET_, mode='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = torchvision.models.efficientnet_b0(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH_))

    # Evaluate the model
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    print('Evaluating the model...')

    # Disable gradient calculation during testing
    progress_bar = tqdm(total=len(test_loader), desc='Test', position=0, leave=True)
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update the progress bar
            progress_bar.update(1)

    # Close the progress bar for the test set
    progress_bar.close()

    # Calculate average test loss and accuracy
    average_test_loss = test_loss / len(test_loader.dataset)
    accuracy = correct / total

    print(f'Test Loss: {average_test_loss:.4f}, Accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    main()
