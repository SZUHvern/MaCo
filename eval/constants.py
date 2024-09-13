from pathlib import Path
import socket

def get_local_ip():
    """
    获取本机IP地址
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

ip = get_local_ip()

MIMIC_DATA_DIR = Path("/path/to/MIMIC-CXR/")
MS_CXR_JSON = Path("/path/to/ms-cxr/MS_CXR_Local_Alignment_v1.0.0.json")
CHESTX_DET10_JSON = Path("/path/to/ChestX-Det10/test.json")
RSNA_DATA_DIR = Path("/path/to/SIIM_ACR_Pneumothorax_and_RSNA_Pneumonia/rsna-pneumonia-detection-challenge/")
PNEUMOTHORAX_DATA_DIR = Path("/path/to/SIIM_ACR_Pneumothorax_and_RSNA_Pneumonia/SIIM ACR Pneumothorax Segmentation Data/")
COVID_RURAL_DATA_DIR = Path("/path/to/covid_rural_annot/")
CHEXLOCALIZE_DATA_DIR = Path("/path/to/chexlocalize/")
CHESTX_DET10_DATA_DIR = Path("/path/to/ChestX-Det10/")


MIMIC_IMG_DIR = MIMIC_DATA_DIR / "MIMIC-224-inter-area/files/"

RSNA_IMG_DIR = RSNA_DATA_DIR / "png_all"
RSNA_CSV = RSNA_DATA_DIR / "stage_2_train_labels.csv"
RSNA_MEDKLIP_CSV = RSNA_DATA_DIR / "MedKLIP_test.csv"

PNEUMOTHORAX_IMG_DIR = PNEUMOTHORAX_DATA_DIR / "dicom-images-train"
PNEUMOTHORAX_ORIGINAL_CSV = PNEUMOTHORAX_DATA_DIR / "train-rle.csv"
PNEUMOTHORAX_MAP_CSV = PNEUMOTHORAX_DATA_DIR / "map.csv"

COVID_RURAL_IMG_DIR = COVID_RURAL_DATA_DIR / "jpgs"
COVID_RURAL_MASK_DIR = COVID_RURAL_DATA_DIR / "pngs_masks"

CHEXLOCALIZE_TEST_IMG_DIR = CHEXLOCALIZE_DATA_DIR / "CheXpert/test"
CHEXLOCALIZE_TEST_JSON = CHEXLOCALIZE_DATA_DIR / "CheXlocalize/gt_segmentations_test.json"
CHEXLOCALIZE_VAL_IMG_DIR = CHEXLOCALIZE_DATA_DIR / "CheXpert/val"
CHEXLOCALIZE_VAL_JSON = CHEXLOCALIZE_DATA_DIR / "CheXlocalize/gt_segmentations_val.json"

CHESTX_DET10_IMG_DIR = CHESTX_DET10_DATA_DIR / "test_data"
