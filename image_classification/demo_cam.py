""""
Grad-CAM visualization
Support for poolformer, deit, resmlp, resnet, swin and convnext
Modifed from: https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py

please install the following packages
`pip install grad-cam timm`

Example command:
python demo_cam.py --image-path ./misc/hammerhead.JPEG --model spanetv2_s18_pure --output-image-path ./misc/hammerhead_cam.JPEG
"""

import argparse
import os
import cv2
import numpy as np
import torch
import timm
import requests

from pytorch_grad_cam import ScoreCAM, GradCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import spanetv2_s18_pure

# ------------- #
# == Reshape == #
# ------------- #
def reshape_transform_resmlp(tensor, height=14, width=14):
    result = tensor.reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_swin(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_transform_metaformer(tensor):
    # (B, H, W, C) -> (B, C, H, W)
    result = tensor.transpose(2, 3).transpose(1, 2)
    return result


# ---------- #
# == Args == #
# ---------- #
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration (enabled by default if available)')
    parser.add_argument(
        '--image-path',
        type=str,
        default=None,
        help='Input image path')
    parser.add_argument(
        '--output-image-path',
        type=str,
        default='./misc/cam_output.png',
        help='Output image path')
    parser.add_argument(
        '--model',
        type=str,
        default='resnet50',
        help='model name')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='scorecam',
                        choices=['gradcam', 'gradcam++',
                                 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam/layercam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args




if __name__ == "__main__": 
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = {"scorecam": ScoreCAM}        

    
    # == Model == #

    model_name = args.model
    if model_name == 'spanetv2_s18_pure':
        model = spanetv2_s18_pure(pretrained=False) # can change different model name
        checkpoint = torch.load('./ckpt/spanetv2_s18_pure_res-scale_Full-ExSPAM.pth', map_location='cpu')

    elif model_name == 'spanetv2_s18_hybrid':
        model = spanetv2_s18_hybrid(pretrained=False) # can change different model name
        checkpoint = torch.load('./ckpt/spanetv2_s18_hybrid_k7_res-scale_Full-ExSPAM.pth', map_location='cpu')


    model.load_state_dict(checkpoint)
    model.to(args.device)
    model.eval()

    target_layers = [model.stages[-1]]   # Choose the target layer you want to compute the visualization for.
                                           # Usually this will be the last convolutional layer in the model.




    # == Input opt. == # 
    img_path = args.image_path
    save_name = args.output_image_path


    # == Transform == #
    if model_name in ['spanetv2_s18_pure', 'spanetv2_s18_hybrid']:
        reshape_transform = reshape_transform_metaformer
    else:
        reshape_transform = None
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imwrite("./misc/cam_input.png", img)


    rgb_img = img[:, :, ::-1]   # BGR -> RGB 
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(args.device)


    # -- We have to specify the target we want to generate the Class Activation Maps for.
    #    If targets is None, the highest scoring category (for every member in the batch) will be used.
    #    You can target specific categories by
    #    targets = [e.g ClassifierOutputTarget(281)]
    #targets = [ClassifierOutputTarget(4)]
    targets = None


    # -- Using the with statement ensures the context is freed, and you can
    #    recreate different CAM objects in a loop.
    cam_algorithm = methods[args.method]

    with cam_algorithm(model=model,
                        target_layers=target_layers,
                        reshape_transform=reshape_transform, 
                        ) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32
        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=args.aug_smooth,
                            eigen_smooth=args.eigen_smooth)

        # -- Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # -- cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)


    cv2.imwrite(save_name, cam_image)

    print()

    


    










