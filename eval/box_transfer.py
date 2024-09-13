import numpy as np
import pandas as pd
import os


def box_transfer(box, w, h, scale):
    """
    Transfer the box from the original image to the resized image
    :param box: box in the original image (x, y, w, h)
    :param w: width of the original image
    :param h: height of the original image
    :param scale: the scale of the resized image
    :return: box in the resized image
    """
    size = (h, w)
    max_dim = max(size)
    max_ind = size.index(max_dim)
    box = np.array(box)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
        box = box * wpercent
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
        box = box * hpercent

    # Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - desireable_size[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - desireable_size[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)

    box[0] = int(np.floor(box[0] + left))
    box[1] = int(np.floor(box[1] + top))
    box[2] = int(np.floor(box[2]))
    box[3] = int(np.floor(box[3]))

    return box.astype(np.int32)


def box2mask(box, w, h):
    """
    Transfer the box to mask
    :param box: box in the original image (x, y, w, h)
    :param w: width of the original image
    :param h: height of the original image
    :return: mask in the original image
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] = 1
    return mask


if __name__ == "__main__":
    import cv2
    df = pd.read_csv("/path/to/ms-cxr/MS_CXR_Local_Alignment_v1.0.0.csv")
    boxs = df[["x", "y", "w", "h"]].values
    w = df["image_width"].values
    h = df["image_height"].values
    scale = 512
    new_boxs = []
    for i in range(len(boxs)):
        new_boxs.append(box_transfer(boxs[i], w[i], h[i], scale))
    new_boxs = np.array(new_boxs)

    ori_data_root = "/path/to/ms-cxr/MS-img/"
    resize_data_root = "/path/to/MIMIC-CXR/MIMIC-512/"
    path = df["path"].values
    di = df["dicom_id"].values

    for i in range(10, len(path)):
        ori_img_path = os.path.join(ori_data_root, path[i])
        ori_img = cv2.imread(ori_img_path)
        resize_img_path = os.path.join(resize_data_root, path[i])
        resize_img = cv2.imread(resize_img_path)
        ori_box = tuple(boxs[i])
        resize_box = tuple(new_boxs[i])
        cv2.rectangle(ori_img, ori_box[:2], (ori_box[0] + ori_box[2], ori_box[1] + ori_box[3]), (0, 0, 255), 7)
        cv2.imwrite("{}_ori.jpg".format(di[i]), ori_img)
        cv2.rectangle(resize_img, resize_box[:2], (resize_box[0] + resize_box[2], resize_box[1] + resize_box[3]), (0, 0, 255), 2)
        cv2.imwrite("{}_resize.jpg".format(di[i]), resize_img)
        break
