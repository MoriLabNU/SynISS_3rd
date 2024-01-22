from skimage.io import imsave, imread
from skimage.color import label2rgb
import numpy as np

# ===========================================================
# Please include import statements needed for your code here.
# START PARTICIPANT CODE
from segment_anything import sam_model_registry
from importlib import import_module
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets.dataset_synapse import SynISS4DatasetTest
import cv2
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
print('GPU (A100: ~3s/img) is much faster than CPU (AMD EPYC 7402: ~30s/img). '
      'But some CUDA version may not be compatible with this Docker image.')

ckpt = './weights/base.pth'
lora_ckpt = './weights/parts.pth'

sam, _ = sam_model_registry['vit_h'](image_size=512, num_classes=3, checkpoint=ckpt, pixel_mean=[0, 0, 0],
                                     pixel_std=[1, 1, 1])
pkg = import_module('sam_lora_image_encoder')
net = pkg.LoRA_Sam(sam, 4).to(device)  # .cuda()
net.load_lora_parameters(lora_ckpt)
net = net.to(device)
multi_mask_output = True
net.eval()

SynISS4_dataset_test = SynISS4DatasetTest()


# END PARTICIPANT CODE
# End of your import statements.
# ===========================================================

def segment(input_image_path, output_mask_path):
    """
    Segment the provided image and save the result in an image file.
    Parameters:
    -----------
    input_image_path: str
        Path to the file that contains the image to be segmented.
    Returns:
    --------
    output_mask_path: str
        Path to the file where the predicted mask should be stored as an image file.
    """
    # ===============================================
    # Place your code below this comment block.
    # Your code must read an image present at the `input_image_path` location. 
    # Your code must generate a numpy array that is the same size as the input image. 
    # The numpy array should be assigned to a variable named `pred_labels`.
    # The numpy array must have a dtype of numpy.uint8 or np.uint8.
    # The array must indicate the class the pixel belongs to by assigning the appropriate class label ID. 
    # Class IDs: shaft = 1, wrist = 2, jaw = 3, background = 0
    # 
    # Example:
    # --------
    # The code below is to ensure that the scripts function and produce output in desired format. 
    # The code is simply creating a copy of sample groundtruth masks. 
    # The test data will NOT contain such groundtruth masks. 
    # 
    # START PARTICIPANT CODE
    # ===============================================

    # dummy predictions - creating predictions using the groundtruth images 
    # please delete this code 
    # ensure that the `pred_labels` variable is populated as per instructions above.
    # =====================EXAMPLE CODE BEGIN==========================
    # gt_rgb = imread(input_image_path.replace("s-", "p-"))
    # pred_labels = np.zeros((gt_rgb.shape[0], gt_rgb.shape[1]), dtype=np.uint8)
    # class_colors = {
    #     1: [255, 214, 0],
    #     2: [138, 0, 0],
    #     3: [49, 205, 49]
    # }
    # for k, v in class_colors.items():
    #     roi = (
    #         (gt_rgb[:,:,0] == v[0]) &
    #         (gt_rgb[:,:,1] == v[1]) &
    #         (gt_rgb[:,:,2] == v[2])
    #     )
    #     pred_labels[roi] = k
    # =====================EXAMPLE CODE END==========================

    sample = SynISS4_dataset_test.get_sample(input_image_path)
    image, case_name, crops = sample['image'], sample['case_name'], sample['crops']  # TODO: confirm the image shape
    slice = image
    inputs = slice.to(device)  # .cuda()
    net.eval()
    width, height = 960, 540
    patch_size = 512
    with torch.no_grad():
        # inputs shape: (1, 3, 512, 512)
        inputs = inputs.unsqueeze(0)
        # print('inputs shape:', inputs.shape)
        outputs = net(inputs, multi_mask_output, patch_size)
        output_masks = outputs['masks']
        out = torch.softmax(output_masks, dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        pred = out
        for i in range(pred.shape[0]):
            temp = pred[i]
            temp = np.vstack((temp[1:], temp[-1]))
            pred[i] = temp
        prediction = pred
        prediction_crops = np.zeros((out.shape[0], width, height))
        prediction_crops = cv2.resize(np.transpose(prediction_crops, (1, 2, 0)), (width, height))
        crop_type = 'resize'
        quarter_height = height // 4
        quarter_width = width // 6
        half_height = height // 2
        third_width = width // 3
        xs_min = [0, 176, 352, 528, 704]
        ys_min = [0, 142, 284]
        crop_size = 256
        if crops is not None:
            for i in range(3):
                for j in range(5):
                    crop = crops[i * 5 + j]['image']
                    # crop shape: (1, 3, 512, 512)
                    crop = crop.unsqueeze(0)
                    # print('crop shape:', crop.shape)
                    crop = crop.to(device)  # .cuda()
                    output_crop = net(crop, multi_mask_output, patch_size)
                    output_masks = output_crop['masks']
                    out = torch.softmax(output_masks, dim=1).squeeze(0)
                    out = out.cpu().detach().numpy()
                    pred = out
                    if crop_type == 'resize':
                        for k in range(pred.shape[0]):
                            temp = pred[k]
                            temp = np.vstack((temp[1:], temp[-1]))
                            pred[k] = temp
                    new_width, new_height = third_width, half_height
                    for k in range(out.shape[0]):
                        if crop_type == 'resize':
                            prediction_crop = cv2.resize(pred[k], (new_width, new_height),
                                                         interpolation=cv2.INTER_NEAREST)
                            prediction_crops[quarter_height * i:quarter_height * i + new_height,
                            quarter_width * j:quarter_width * j + new_width, k] += prediction_crop
                        elif crop_type == '256':
                            prediction_crop = cv2.resize(pred[k], (crop_size, crop_size),
                                                         interpolation=cv2.INTER_NEAREST)
                            prediction_crops[ys_min[i]:ys_min[i] + crop_size, xs_min[j]:xs_min[j] + crop_size,
                            k] += prediction_crop
                        else:
                            raise NotImplementedError
    prediction = cv2.resize(np.transpose(prediction, (1, 2, 0)), (width, height))
    prediction = prediction_crops + prediction
    prediction = np.argmax(prediction, axis=2)
    prediction = cv2.resize(prediction, (width, height), interpolation=cv2.INTER_NEAREST)
    pred_labels = prediction.astype(np.uint8)

    # ================================================
    # END PARTICIPANT CODE
    # End of your code. DO NOT modify the code beyond this point.
    # ================================================

    # convert the numpy array to RGB image and save it to file
    imsave(output_mask_path, (label2rgb(pred_labels, colors=["gold", "darkred", "limegreen"]) * 255.0).astype(np.uint8))
    
    return