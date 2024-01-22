import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


class SynISS4DatasetTest:
    def __init__(self, transform=A.Compose([
        A.Resize(width=512, height=512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])):
        # print('using SynISS4_dataset_test')
        self.transform = transform  # using transform in torch!
        self.split = 'test'

    def get_sample(self, input_image_path):
        image_path = input_image_path
        image = cv2.imread(image_path).astype('uint8')

        sample = {'image': image}
        if self.transform:
            sample = self.transform(image=image)
        if self.split != 'train':
            height, width = image.shape[:2]
            assert height == 540 and width == 960
            # print(height, width)  # 540 960
            crop_type = 'resize'
            quarter_height = height // 4
            quarter_width = width // 6
            half_height = height // 2
            third_width = width // 3
            xs_min = [0, 176, 352, 528, 704]
            ys_min = [0, 142, 284]
            crop_size = 256
            crops = []
            for i in range(3):
                for j in range(5):
                    if crop_type == 'resize':
                        x_min = j * quarter_width
                        x_max = x_min + third_width
                        y_min = i * quarter_height
                        y_max = y_min + half_height
                    elif crop_type == '256':
                        x_min = xs_min[j]
                        x_max = x_min + crop_size
                        y_min = ys_min[i]
                        y_max = y_min + crop_size
                    else:
                        raise NotImplementedError

                    x_max = min(x_max, width)
                    y_max = min(y_max, height)

                    crop = {'image': image[y_min:y_max, x_min:x_max, :]}
                    if self.transform:
                        crop = self.transform(image=crop['image'])
                    crops.append(crop)
            sample['crops'] = crops

        sample['case_name'] = input_image_path.split('/')[-1]
        return sample
