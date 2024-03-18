### Docker container settings
```
--entrypoint -v [PROJECT_PATH]:/opt/project -v [DATASET_PATH]:/opt/project/datasets/food-101:ro --rm --gpus all
```

### Dataset
https://huggingface.co/datasets/food101

To run inference for now create a validate folder in the dataset folder containing validation images.