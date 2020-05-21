import fire
import getpass
from datasets.chest_xray_pneumonia_dataset import ChestXRayPneumoniaDataset


def main(usr, project, dst_path):
    # Get platform password
    platform_psw = getpass.getpass(f"Password [{usr}]")

    if project == 'chest_xray_pneumonia':
        dataset = ChestXRayPneumoniaDataset(path=dst_path)
    elif project == 'nih_cx38':
        raise ValueError('NIHCC download not implemented')
    else:
        raise ValueError(f'{project} not a valid value, select between chest_xray_pneumonia, covid-chestxray and nih_cx38')
    
    dataset.download_from_platform(usr, platform_psw, dst_path)

if __name__ == "__main__":
    fire.Fire(main)