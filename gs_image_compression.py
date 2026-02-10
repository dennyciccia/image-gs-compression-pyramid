import shutil
import cv2
import os
import subprocess
import numpy as np
from skimage.metrics import structural_similarity as ssim


MEDIA_PATH = "image-gs/media/images"
N_GAUSSIANS = 10000
MIN_RES = 1024
IMG_NAME = "city"
IMG_FORMAT = "jpg"
PARAMETERS = f"ng-{N_GAUSSIANS}_mr-{MIN_RES}"
EXP_NAME = f"{IMG_NAME}_{PARAMETERS}"


class Image:
    def __init__(self):
        self.__image_obj = None
        self.__path = ""

    @property
    def image_obj(self):
        return self.__image_obj

    @image_obj.setter
    def image_obj(self, value):
        self.__image_obj = value

    @property
    def w(self):
        return self.image_obj.shape[1]
    
    @property
    def h(self):
        return self.image_obj.shape[0]
    
    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, value):
        self.__path = value

    @property
    def name(self):
        return '.'.join(self.path.split('/')[-1].split('.')[:-1])
    
    @property
    def base_name(self):
        return f"level_{self.level}_{self.w}x{self.h}"
    
    @property
    def level(self):
        return int(self.name[6])
    
    def read(self, path):
        self.path = path
        self.image_obj = cv2.imread(self.path)
        if self.image_obj is None:
            raise FileNotFoundError("File not found in: " + path)
        
    def save(self, path):
        self.path = path
        cv2.imwrite(self.path, self.image_obj)

    def get_path_latest_levels(self, level):
        return '/'.join(self.path.split('/')[level:])
    
    def get_path_for_image_gs(self):
        return self.get_path_latest_levels(-4)
    
    def get_file_size(self):
        return os.path.getsize(self.path)


def create_image_pyramid(image_path, output_dir, min_res):
    # Load starting image
    starting_img = Image()
    try:
        starting_img.read(image_path)
    except FileNotFoundError:
        print("Error: Image not found in: " + image_path)
        return None

    pyramid = [starting_img]
    
    # Halve until minimun resolution is reached
    while True:
        height, width = pyramid[-1].image_obj.shape[:2]
        #new_w, new_h = width // 2, height // 2
        new_h = height // 2
        new_w = round((width / height) * new_h)
        
        if new_h < min_res:
            break
            
        # Resize
        new_image_lower_res = Image()
        new_image_lower_res.image_obj = cv2.resize(pyramid[-1].image_obj, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pyramid.append(new_image_lower_res)
        
    # List reverse to start from minimul resolution -> [min_res, ..., max_res]
    pyramid = pyramid[::-1]

    # Create output directory if it does not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

    # Save images of differet levels
    for i, level_img in enumerate(pyramid):
        level_img.save(f"{output_dir}/level_{i}_{level_img.w}x{level_img.h}.png")
        print(f"Saved: {level_img.path}")

    return pyramid


def fit_gaussians(img_to_fit, n_gaussians, render_height, exp_name, output_dir, name_suffix = "fit_upsample"):
    # Run Image-GS
    os.chdir('image-gs')
    subprocess.run(['python', 'main.py', f'--input_path={img_to_fit.get_path_for_image_gs()}', f'--exp_name=test/{exp_name}/{img_to_fit.name}', f'--num_gaussians={n_gaussians}', '--quantize', '--save_image_format=png', '--save_plot_format=png'])
    subprocess.run(['python', 'main.py', f'--input_path={img_to_fit.get_path_for_image_gs()}', f'--exp_name=test/{exp_name}/{img_to_fit.name}', f'--num_gaussians={n_gaussians}', '--quantize', '--eval', f'--render_height={render_height}', '--save_image_format=png', '--save_plot_format=png'])
    os.chdir('..')

    # Read fitted image and save a copy in the media folder
    img_fitted = Image()
    input_path = f"image-gs/results/test/{exp_name}/{img_to_fit.name}/num-{n_gaussians}_inv-scale-5.0_bits-16-16-16-16_top-10_g-0.3_l1-1.0_l2-0.0_ssim-0.1_decay-1-10.0_prog/eval/"
    input_fn = os.listdir(input_path)[0]
    img_fitted.read(input_path + input_fn)
    output_path = f"{output_dir}/{img_to_fit.base_name}_{name_suffix}.png"
    img_fitted.save(output_path)
    print("Fitted image saved in: " + img_fitted.path)

    return img_fitted


def compute_residual(img_original, img_upsample, output_dir):
    # Resize if both images don't have same dimensions
    if img_original.h != img_upsample.h or img_original.w != img_upsample.w:
        img_upsample.image_obj = cv2.resize(img_upsample.image_obj, (img_original.w, img_original.h), interpolation=cv2.INTER_CUBIC)
    
    diff = Image()

    # Use float32 to avoid overflow during subtraction
    raw_diff = img_original.image_obj.astype(np.float32) - img_upsample.image_obj.astype(np.float32)
    offset_diff = (raw_diff / 2.0) + 128.0
    diff.image_obj = np.clip(offset_diff, 0, 255).astype(np.uint8)
    
    # Save in media folder
    output_path = f"{output_dir}/{img_original.name}_diff.png"
    diff.save(output_path)
    print(f"Residual saved in: {diff.path}")

    return diff


def compute_sum(img_detail, img_base, output_dir):
    # Resize because both images must have same dimensions
    if img_detail.h != img_base.h or img_detail.w != img_base.w:
        img_base_upsample = Image()
        img_base_upsample.image_obj = cv2.resize(img_base.image_obj, (img_detail.w, img_detail.h), interpolation=cv2.INTER_CUBIC)
    else:
        img_base_upsample = img_base

    # Compute sum by restoring original range
    base_f = img_base_upsample.image_obj.astype(np.float32)
    detail_f = img_detail.image_obj.astype(np.float32)
    real_difference = (detail_f - 128.0) * 2.0
    summed_f = base_f + real_difference
    img_result = Image()
    img_result.image_obj = np.clip(summed_f, 0, 255).astype(np.uint8)

    # Save in media folder
    output_path = f"{output_dir}/{img_detail.base_name}_sum.png"
    img_result.save(output_path)
    print(f"Sum saved in: {img_result.path}")

    return img_result


def get_ssim_score(img1, img2):
    # Resize if needed
    if img1.image_obj.shape[:2] != img2.image_obj.shape[:2]:
        img2.image_obj = cv2.resize(img2.image_obj, (img1.w, img1.h), interpolation=cv2.INTER_CUBIC)

    # Conversion to grey scale (SSIM works on structure/brightness)
    gray1 = cv2.cvtColor(img1.image_obj, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2.image_obj, cv2.COLOR_BGR2GRAY)

    score = ssim(gray1, gray2)
    
    return score


def get_psnr_score(img1, img2):
    # Resize if needed
    if img1.image_obj.shape[:2] != img2.image_obj.shape[:2]:
        img2.image_obj = cv2.resize(img2.image_obj, (img1.w, img1.h), interpolation=cv2.INTER_CUBIC)

    score = cv2.PSNR(img1.image_obj, img2.image_obj) # dB
    
    return score


def clear_image_gs_result_dir(exp_name):
    shutil.rmtree(f"image-gs/results/test/{exp_name}")


def main():
    gs_fitting = []
    sums = []
    diffs = [None]

    print("Building pyramid ...")
    pyramid = create_image_pyramid(f"{MEDIA_PATH}/{IMG_NAME}/{IMG_NAME}.{IMG_FORMAT}", f"{MEDIA_PATH}/{IMG_NAME}/{PARAMETERS}", MIN_RES)
    if pyramid is None:
        exit(1)
    print()

    print("Pyramid levels:")
    for i, level_img in enumerate(pyramid):
        print(f"Level {i}: Resolution {level_img.w}x{level_img.h}")
    print()

    # Level 0
    print("Fitting level 0 image using Image-GS ...")
    sums.append(fit_gaussians(pyramid[0], N_GAUSSIANS, pyramid[1].h, EXP_NAME, f"{MEDIA_PATH}/{IMG_NAME}/{PARAMETERS}"))
    gs_fitting.append(sums[0])
    print()

    # Level 1 ... n-1
    for i in range(1, len(pyramid)):
        print(f"---------- Level {i} ----------\n")

        print("Creating a separated fit of image at current level ...")
        gs_fitting.append(fit_gaussians(pyramid[i], N_GAUSSIANS, pyramid[i].h, EXP_NAME, f"{MEDIA_PATH}/{IMG_NAME}/{PARAMETERS}", "separated_fit"))
        print()

        print("Computing residual ...")
        diffs.append(compute_residual(pyramid[i], sums[i-1], f"{MEDIA_PATH}/{IMG_NAME}/{PARAMETERS}"))
        print()

        print("Fitting residual using Image-GS ...")
        diffs[i] = fit_gaussians(diffs[i], N_GAUSSIANS, pyramid[i+1 if i+1 < len(pyramid) else i].h, EXP_NAME, f"{MEDIA_PATH}/{IMG_NAME}/{PARAMETERS}", "diff_fit_upsample")
        print()

        print("Computing sum ...")
        sums.append(compute_sum(diffs[i], sums[i-1], f"{MEDIA_PATH}/{IMG_NAME}/{PARAMETERS}"))
        print()
    
    # Measurements w.r.t. pyramid image at level i
    print("Measuring ssim, psnr and compression rate ...")
    with open(f"{MEDIA_PATH}/{IMG_NAME}/{PARAMETERS}/measurements.csv", 'w') as fd:
        print("level,ssim_sums,psnr_sums,cr_sums,ssim_fitting,psnr_fitting,cr_fitting,", file=fd)
        for i in range(len(pyramid)):
            print(i, file=fd, end=',')
            print(round(get_ssim_score(sums[i], pyramid[i]), 2), file=fd, end=',')
            print(round(get_psnr_score(sums[i], pyramid[i]), 2), file=fd, end=',')
            print(round((sums[i].get_file_size() / pyramid[i].get_file_size()) * 100, 2), file=fd, end=',')
            print(round(get_ssim_score(gs_fitting[i], pyramid[i]), 2), file=fd, end=',')
            print(round(get_psnr_score(gs_fitting[i], pyramid[i]), 2), file=fd, end=',')
            print(round((gs_fitting[i].get_file_size() / pyramid[i].get_file_size()) * 100, 2), file=fd, end='\n')
    print()
    
    print("Compression completed")


if __name__ == "__main__":
    main()