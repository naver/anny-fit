import torch
import numpy as np
import cv2

def perspective_projection(x, K):
    """
    This function computes the perspective projection of a set of points assuming the extrinsinc params have already been applied
    Args:
        - x [bs,N,3]: 3D points
        - K [bs,3,3]: Camera instrincs params
    """
    # Apply perspective distortion
    y = x / x[:, :, -1].unsqueeze(-1)  # (bs, N, 3)

    # Apply camera intrinsics
    y = torch.einsum('bij,bkj->bki', K, y)  # (bs, N, 3)

    return y[:, :, :2]

def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None):
    """
    Given image size (tuple), fov and principal point coordinates, return K the camera parameter matrix
    """
    K = torch.eye(3)
    # Get focal length.
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0,0], K[1,1] = focal
    # Set principal point
    if p_x is not None and p_y is not None:
            K[0,-1], K[1,-1] = p_x * img_size[0], p_y * img_size[1]
    else:
            K[0,-1], K[1,-1] = img_size//2
    return K


def get_focalLength_from_fieldOfView(fov=60, img_size=512):
    """
    Compute the focal length of the camera lens by assuming a certain FOV for the entire image
    Args:
        - fov: float, expressed in degree
        - img_size: int
    Return:
        focal: float
    """
    focal = img_size / (2 * np.tan(np.radians(fov) /2))
    return focal


class GradientPreservingClamp(torch.autograd.Function):
    """
    Clamp values but preserve gradient pushing towards the valid region
    """
    @staticmethod
    def forward(ctx, x, low, high):
        ctx.save_for_backward(x)
        ctx.low = low
        ctx.high = high
        return torch.clamp(x, low, high)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        low = ctx.low
        high = ctx.high
        # Ignore gradients pushing further out of the valid range when x is outside it
        grad_input = grad_output * (~(((x <= low) & (grad_output > 0)) | ((x >= high) & (grad_output < 0)))).to(dtype=grad_output.dtype)
        return grad_input, None, None
    
def clamp_but_preserve_gradients(x, low, high):
    """
    Clamp values but preserve gradient pushing towards the valid region
    """
    return GradientPreservingClamp.apply(x, low, high)

def keypointscrop_to_fullimage(keypoints_crop, bbox, resized_crop_size=256):
    """
    Transforms 2D keypoints from a resized crop's coordinates to the
    original full image coordinates.

    Args:
        keypoints_crop (torch.Tensor): Keypoints in the resized crop space.
                                       Shape: (num_keypoints, 2) or (num_keypoints, 3).
        bbox (torch.Tensor): Bounding box of the crop in the original image,
                             in [x_min, y_min, width, height] format.
        resized_crop_size (int): The size the crop was resized to (e.g., 256).

    Returns:
        torch.Tensor: Keypoints in the original image space.
    """
    x_min, y_min, bbox_width, bbox_height = bbox

    # Calculate the scale factor from the resized crop to the original bbox size
    scale_x = bbox_width / resized_crop_size
    scale_y = bbox_height / resized_crop_size

    # Create a copy to avoid modifying the original tensor
    keypoints_img = keypoints_crop.clone()

    # Apply the scale and then translate
    keypoints_img[:, 0] = keypoints_img[:, 0] * scale_x + x_min
    keypoints_img[:, 1] = keypoints_img[:, 1] * scale_y + y_min

    return keypoints_img

def norm_kps_to_fullimage(keypoints_norm, bbox, image_size):
    """
    Transforms 2D keypoints from normalized crop coordinates [0, 1] to
    original full image coordinates and updates the confidence for points outside the image.

    Args:
        keypoints_norm (torch.Tensor): Normalized keypoints in the crop space.
                                       Shape: (num_keypoints, 3) for (x, y, conf).
        bbox (torch.Tensor): Bounding box of the crop in the original image,
                             in [x_min, y_min, width, height] format.
        image_size (tuple): A tuple (height, width) of the original image.

    Returns:
        torch.Tensor: Keypoints in the original image space, with confidence
                      set to 0 for points outside the image.
    """
    x_min, y_min, bbox_width, bbox_height = bbox
    img_height, img_width = image_size
    
    # Create a copy to avoid modifying the original tensor
    keypoints_img = keypoints_norm.clone()

    # Denormalize by scaling with the bbox dimensions and translating
    keypoints_img[:, 0] = keypoints_img[:, 0] * bbox_width + x_min
    keypoints_img[:, 1] = keypoints_img[:, 1] * bbox_height + y_min
    
    # Create a mask for keypoints that are inside the image boundaries
    mask_x = (keypoints_img[:, 0] >= 0) & (keypoints_img[:, 0] < img_width)
    mask_y = (keypoints_img[:, 1] >= 0) & (keypoints_img[:, 1] < img_height)
    valid_mask = mask_x & mask_y

    # Set confidence to zero for keypoints outside the image
    keypoints_img[:, 2] = keypoints_img[:, 2] * valid_mask
    keypoints_img[:, 2] = keypoints_img[:, 2].clamp(0, 1)  # Ensure confidence is in [0, 1]

    return keypoints_img

def dense_kps_to_fullimage(keypoints_norm, bbox_xyxy):
    """Project normalized dense keypoints ([-0.5, 0.5] crop coords) to full-image pixels."""
    bbox_center = ((bbox_xyxy[:, :2] + bbox_xyxy[:, 2:]) / 2.0).unsqueeze(1)  # (bs, 1, 2)
    bbox_wh = bbox_xyxy[:, 2:] - bbox_xyxy[:, :2]
    longest_side = bbox_wh.max(dim=-1)[0]
    scale = (longest_side / 200.0).unsqueeze(1).unsqueeze(1)  # (bs, 1, 1)
    keypoints_img = keypoints_norm.clone()
    keypoints_img[:, :, :2] = 200.0 * keypoints_norm[:, :, :2] * scale + bbox_center
    return keypoints_img

# TODO: figure out relative imports to avoid repeating these

def coco19tococo17(pose3d):
    # https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html
    # https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/docs/zh_cn/dataset_zoo/3d_body_keypoint.md
    coco17_indices = np.array([1, 15, 17, 16, 18, 3, 9, 4, 10, 5, 11, 6, 12, 7, 13, 8, 14])
    return pose3d[coco17_indices, ...]


import numpy as np
from itertools import product

def l2_error(j1, j2):
    return np.linalg.norm(j1 - j2, 2)

def get_bbx_overlap(p1, p2, imgpath, baseline=None):
    min_p1 = np.min(p1, axis=0)
    min_p2 = np.min(p2, axis=0)
    max_p1 = np.max(p1, axis=0)
    max_p2 = np.max(p2, axis=0)

    bb1 = {}
    bb2 = {}

    bb1['x1'] = min_p1[0]
    bb1['x2'] = max_p1[0]
    bb1['y1'] = min_p1[1]
    bb1['y2'] = max_p1[1]
    bb2['x1'] = min_p2[0]
    bb2['x2'] = max_p2[0]
    bb2['y1'] = min_p2[1]
    bb2['y2'] = max_p2[1]

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = max(0, x_right - x_left + 1) * \
        max(0, y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou

def match_2d_greedy(
        pred_kps,
        gtkp,
        valid_mask,
        imgPath=None,
        baseline=None,
        iou_thresh=0.05,
        valid=None,
        ind=-1):
    '''
    matches groundtruth keypoints to the detection by considering all possible matchings.
    :return: best possible matching, a list of tuples, where each tuple corresponds to one match of pred_person.to gt_person.
            the order within one tuple is as follows (idx_pred_kps, idx_gt_kps)
    '''
    predList = np.arange(len(pred_kps))
    gtList = np.arange(len(gtkp))
    # get all pairs of elements in pred_kps, gtkp
    # all combinations of 2 elements from l1 and l2
    combs = list(product(predList, gtList))

    errors_per_pair = {}
    errors_per_pair_list = []
    for comb in combs:
        vmask = valid_mask[comb[1]]
        assert vmask.sum()>0, print('no valid points')
        errors_per_pair[str(comb)] = l2_error(
            pred_kps[comb[0]][vmask, :2], gtkp[comb[1]][vmask, :2])
        errors_per_pair_list.append(errors_per_pair[str(comb)])

    gtAssigned = np.zeros((len(gtkp),), dtype=bool)
    opAssigned = np.zeros((len(pred_kps),), dtype=bool)
    errors_per_pair_list = np.array(errors_per_pair_list)

    bestMatch = []
    excludedGtBecauseInvalid = []
    falsePositiveCounter = 0
    while np.sum(gtAssigned) < len(gtAssigned) and np.sum(
            opAssigned) + falsePositiveCounter < len(pred_kps):
        found = False
        falsePositive = False
        while not(found):
            if sum(np.inf == errors_per_pair_list) == len(
                    errors_per_pair_list):
                print('something went wrong here')

            minIdx = np.argmin(errors_per_pair_list)
            minComb = combs[minIdx]
            # compute IOU
            iou = get_bbx_overlap(
                pred_kps[minComb[0]], gtkp[minComb[1]], imgPath, baseline)
            # if neither prediction nor ground truth has been matched before and iou
            # is larger than threshold
            if not(opAssigned[minComb[0]]) and not(
                    gtAssigned[minComb[1]]) and iou >= iou_thresh:
                #print(imgPath + ': found matching')
                found = True
                errors_per_pair_list[minIdx] = np.inf
            else:
                errors_per_pair_list[minIdx] = np.inf
                # if errors_per_pair_list[minIdx] >
                # matching_threshold*headBboxs[combs[minIdx][1]]:
                if iou < iou_thresh:
                    #print(
                    #   imgPath + ': false positive detected using threshold')
                    found = True
                    falsePositive = True
                    falsePositiveCounter += 1

        # if ground truth of combination is valid keep the match, else exclude
        # gt from matching
        if not(valid is None):
            if valid[minComb[1]]:
                if not falsePositive:
                    bestMatch.append(minComb)
                    opAssigned[minComb[0]] = True
                    gtAssigned[minComb[1]] = True
            else:
                gtAssigned[minComb[1]] = True
                excludedGtBecauseInvalid.append(minComb[1])

        elif not falsePositive:
            # same as above but without checking for valid
            bestMatch.append(minComb)
            opAssigned[minComb[0]] = True
            gtAssigned[minComb[1]] = True

    bestMatch = np.array(bestMatch)
    # add false positives and false negatives to the matching
    # find which elements have been successfully assigned
    opAssigned = []
    gtAssigned = []
    for pair in bestMatch:
        opAssigned.append(pair[0])
        gtAssigned.append(pair[1])
    opAssigned.sort()
    gtAssigned.sort()

    falsePositives = []
    misses = []

    # handle false positives
    opIds = np.arange(len(pred_kps))
    # returns values of oIds that are not in opAssigned
    notAssignedIds = np.setdiff1d(opIds, opAssigned)
    for notAssignedId in notAssignedIds:
        falsePositives.append(notAssignedId)
    gtIds = np.arange(len(gtList))
    # returns values of gtIds that are not in gtAssigned
    notAssignedIdsGt = np.setdiff1d(gtIds, gtAssigned)

    # handle false negatives/misses
    for notAssignedIdGt in notAssignedIdsGt:
        if not(valid is None):  # if using the new matching
            if valid[notAssignedIdGt]:
                misses.append(notAssignedIdGt)
            else:
                excludedGtBecauseInvalid.append(notAssignedIdGt)
        else:
            misses.append(notAssignedIdGt)

    return bestMatch, falsePositives, misses  # tuples are (idx_pred_kps, idx_gt_kps)

def solvePnP(pose3d, pose2d, K):
    """
    Solve the PnP problem with Ransac
    Args:
        - pose3d: np.array [bs, N, 3]
        - pose2d: np.array [bs, N, 2]
        - K: np.array [3, 3]
    Return:
        - rvec: np.array [bs, 3] axis-angle representation
        - tvec: np.array [bs, 3]
    """
    bs = pose3d.shape[0]
    rvec_final = torch.zeros((bs, 3), dtype=torch.float32)
    tvec_final = torch.zeros((bs, 3), dtype=torch.float32)
    cameraMatrix = K[0].detach().cpu().numpy().copy().astype(np.float64)
    for p_idx in range(bs):
        objectPoints = pose3d[p_idx].detach().cpu().numpy().copy().reshape(-1, 1, 3).astype(np.float64)
        imagePoints  = pose2d[p_idx].detach().cpu().numpy().copy().reshape(-1, 1, 2).astype(np.float64)
        # 1) Bootstrap with EPNP (no initial guess)
        _, rvec_1, tvec_1 = cv2.solvePnP(
            objectPoints=objectPoints,
            imagePoints=imagePoints,
            cameraMatrix=cameraMatrix,
            distCoeffs=np.zeros((5,1)),
            flags=cv2.SOLVEPNP_EPNP,
            useExtrinsicGuess=False
        )
        # 2) Refine with ITERATIVE, using the EPNP output as the guess
        _, rvec_2, tvec_2 = cv2.solvePnP(
            objectPoints=objectPoints,
            imagePoints=imagePoints,
            cameraMatrix=cameraMatrix,
            distCoeffs=np.zeros((5,1)),
            rvec=rvec_1,
            tvec=tvec_1,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        rvec_final[p_idx] = torch.from_numpy(rvec_2.reshape(3).astype(np.float32))
        tvec_final[p_idx] = torch.from_numpy(tvec_2.reshape(3).astype(np.float32))
    return rvec_final.to(pose3d.device), tvec_final.to(pose3d.device)

def rh_to_text(rh_number, name="age"):
    if name == "age":
        # 3 is originally called baby but called it toddler to have a more neutral start
        rh_cats = {0: 'adult', 1:'teenager', 2: 'kid', 3: 'toddler'}
    else:
        rh_cats = {0: 'male', 1: 'female'}
    return rh_cats.get(rh_number, "unknown")
    

import re
import roma
import anny
import torch

AGE_MAPPING = {'baby': 0.0, 'toddler': 0.1, 'kid': 0.330, 'child': 0.330, 'teenager': 0.500, 'teen': 0.500, 'adult': 0.660, 'elder': 0.999, "senior": 0.999}
GENDER_MAPPING = {'male': 0, 'female': 1, 'neutral': 0.5, 'unknown': 0.5}
    
ANNY_MAPPING = {
    'age': AGE_MAPPING,
    'gender': GENDER_MAPPING,
}

def map_prediction(pred, map_dict=None, default=-1):
    # handle predictions that don't exactly match the keys in the mapping dict
    pred = pred.lower().strip()
    if map_dict is None:
        return pred
    for key in map_dict:
        if re.search(rf'\b{key}\b', pred):
            return map_dict[key]
    return default

def attributes2shape(person_attributes, model_type='anny', default=-1.0):
    if model_type != 'anny':
        # only anny is supported
        return None
    # convert person attributes to values in 0-1 range for anny shape
    mapped_attributes = {}
    age_name = person_attributes.get('age')
    age = map_prediction(age_name, ANNY_MAPPING['age'], default=default)

    gender_name = person_attributes.get('gender')
    gender = map_prediction(gender_name, ANNY_MAPPING['gender'], default=default)

    mapped_attributes['age'] = age
    mapped_attributes['gender'] = gender

    return mapped_attributes