
import cv2
from pathlib import Path
import numpy as np


def histogram_equalization(rgb_image):
    """
    Histogram equalization is a method in image processing of contrast adjustment using the image's histogram.
    This method usually increases the global contrast of many images, especially when the usable data of the image is
    represented by close contrast values. Through this adjustment, the intensities can be better distributed on the
    histogram. This allows for areas of lower local contrast to gain a higher contrast.
    """
    img_yuv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


    return img_output


def clahe(rgb_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) is a variant of adaptive histogram equalization in which
    the contrast amplification is limited. This method is useful in improving the local contrast and enhancing the
    details in the images.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    

    img_yuv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output


if __name__ == "__main__":
    save_folder = 'image_enhancing_results/'
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    # Read the image
    images = [
        'database/expertv3_3/3_scenario_per_route_40_routes_per_file_all_scenarioss/routes_training/random_weather_seed_111/data/seed_111/0_route0_route0_05_02_12_42_38_05_02_12_42_41/rgb_0/0068.jpg',
        'database/expertv3_3/3_scenario_per_route_40_routes_per_file_all_scenarioss/routes_training/random_weather_seed_111/data/seed_111/0_route0_route2_05_02_13_07_44_05_02_13_07_45/rgb_0/0013.jpg',
    ]

    
    for idx, image in enumerate(images):

        # read rgb image
        bgr_image = cv2.imread(image)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


        # Apply histogram equalization
        equalized_image = histogram_equalization(rgb_image)

        # Apply CLAHE
        clahe_image = clahe(rgb_image)

        # concatenate the images below each other
        concatenated_image = np.concatenate((rgb_image, equalized_image, clahe_image), axis=0)


        # Save the images
        concatenated_image = cv2.cvtColor(concatenated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{save_folder}/{idx}_concatenated_image.jpg', concatenated_image)
