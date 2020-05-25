import numpy as np
import fire
import torch

from models import Resnet34
from datasets import COVIDChestXRayDataset


def covid19_classification(image_path, model_checkpoint_path, image_size=256):
    """
    Given an image and a model checkpoint, this function returns a boolean 
    indicating if the x-ray is classified as COVID19 or not.

    Parameters
    ----------
    image_path : str
    model_checkpoint_path : str
    image_size : int, optional

    Returns
    -------
    bool
        COVID19 classification
    """
    im = COVIDChestXRayDataset._load_image(image_path, image_size)
    loaded_model = load_model_for_inference(model_checkpoint_path)
    preds = inference(im, loaded_model)
    softmax_prediction = preds[0][0].numpy()
    hard_prediction = softmax_prediction > 0.5
    return softmax_prediction, hard_prediction


def inference(image_data, model):
    image_data_batch = np.expand_dims(image_data, axis=0)
    image_tensor = torch.from_numpy(image_data_batch)

    with torch.no_grad():
        predictions = model.forward(image_tensor)

    return predictions


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
    fire.Fire(covid19_classification)
