import numpy as np
import fire
import torch

from models import Resnet34
from datasets import COVIDChestXRayDataset


def main(image_path, model_checkpoint_path, size=128):
    im = COVIDChestXRayDataset._load_image(image_path, size)
    loaded_model = load_model_for_inference(model_checkpoint_path)
    return inference(im, loaded_model)


def inference(image_data, model):
    image_data_batch = np.expand_dims(image_data, axis=0)
    predictions = model(image_data_batch)
    return predictions[0]


def load_model_for_inference(model_checkpoint_path):
    """
    Load model from a checkpoint saved during training, following the recommendations
    by the PyTorch team.

    References
    ----------
    https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended
    """
    model = Resnet34()
    model.load_state_dict(torch.load(model_checkpoint_path))

    # Set up model in eval mode so that batch norm and similar layers are
    # properly set up ()
    model.eval()

    return model


if __name__ == "__main__":
    fire.Fire(main)
