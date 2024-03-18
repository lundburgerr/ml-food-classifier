import torch
import torchvision
import os
from PIL import Image

from train import CustomDataset, PATH_TO_DATASET_, MODEL_PATH_


def main():
    transform = CustomDataset.gen_transforms()
    model = torchvision.models.efficientnet_b0(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH_))
    device = torch.device('cuda')
    model.to(device)
    model.eval()

    classes_file = os.path.join(PATH_TO_DATASET_, 'meta', 'classes.txt')
    with open(classes_file, 'r') as f:
        classes = f.read().splitlines()

    # Preprocess the input batch of images
    image_paths = [f'{PATH_TO_DATASET_}/validate/apple_pie.jpg', f'{PATH_TO_DATASET_}/validate/chocolate_cake.jpg',
                   f'{PATH_TO_DATASET_}/validate/pad_thai.jpg']
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        images.append(image)
    images = torch.stack(images)  # Convert list of tensors to a single tensor (batch)
    images = images.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(images)

    # Interpret the model outputs to get the predictions for each image in the batch
    predicted_classes = torch.argmax(outputs, dim=1).tolist()

    # Get the predicted class labels
    predicted_labels = [classes[class_idx] for class_idx in predicted_classes]
    print("Predicted labels:", predicted_labels)


if __name__ == '__main__':
    main()
