import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Optional

arm_mask_codes = [31, 34, 35, 39, 40, 41, 42, 43, 44, 45, 46]
table_mask_codes = [48, 52]
bg_mask_codes = [10, 55]
scene_bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]


def clean_mask(mask):
    for a in arm_mask_codes:
        mask[mask == a] = 0
    for a in table_mask_codes:
        mask[mask == a] = 1
    return mask

def keep_valid_pcd(pc, other, scene_bounds):
    x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
    inv_pnt = (  # invalid points
        (pc[:, 0] < x_min)
        | (pc[:, 0] > x_max)
        | (pc[:, 1] < y_min)
        | (pc[:, 1] > y_max)
        | (pc[:, 2] < z_min)
        | (pc[:, 2] > z_max)
        | np.isnan(pc[:, 0])
        | np.isnan(pc[:, 1])
        | np.isnan(pc[:, 2])
    )
    return pc[~inv_pnt], tuple([x[~inv_pnt] for x in other])


def normalize_within_bounds(pc, scene_bounds):
    x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
    pc = pc.copy()
    pc[:, 0] = (pc[:, 0] - x_min) / (x_max - x_min)
    pc[:, 1] = (pc[:, 1] - y_min) / (y_max - y_min)
    pc[:, 2] = (pc[:, 2] - z_min) / (z_max - z_min)
    return pc


def gripper_pose_2_frame(gripper_pose, scale=0.25):
    t = normalize_within_bounds(gripper_pose[None, :3], scene_bounds).flatten()
    dcm = Rotation.from_quat(gripper_pose[3:]).as_matrix()
    dcm *= scale
    return t, dcm[0, :] + t, dcm[1, :] + t, dcm[2, :] + t


def assemble_point_cloud(m):
    pcd = []
    rgb = []
    mask = []
    for c in CAMERAS:
        pcd.append(m[f"{c}_point_cloud"].reshape(-1, 3))
        rgb.append(m[f"{c}_rgb"].reshape(-1, 3))
        mask.append(m[f"{c}_mask"].reshape(-1, 3))
    pcd = np.concatenate(pcd)
    rgb = np.concatenate(rgb)
    mask = np.concatenate(mask)
    pcd, (rgb, mask) = keep_valid_pcd(pcd, (rgb, mask), scene_bounds)
    pcd = normalize_within_bounds(pcd, scene_bounds)
    return pcd, rgb, clean_mask(mask[:, 0])


def transfer_gripper_pose(
    pcd1,
    pcd2,
    mask1,
    mask2,
    object_id_1,
    object_id_2,
    gripper_pose_1,
    origin_frame_scale=0.25,
):
    obj_pts_1 = pcd1[mask1 == object_id_1]
    obj_pts_2 = pcd2[mask2 == object_id_2]

    min_len = min(len(obj_pts_1), len(obj_pts_2))

    if len(obj_pts_1) > min_len:
        obj_pts_1 = obj_pts_1[np.random.permutation(len(obj_pts_1))[:min_len]]

    if len(obj_pts_2) > min_len:
        obj_pts_2 = obj_pts_2[np.random.permutation(len(obj_pts_2))[:min_len]]

    T, _, _ = icp(obj_pts_1, obj_pts_2)

    # translation
    pose1_t = normalize_within_bounds(gripper_pose_1[None, :3], scene_bounds).flatten()
    new_t = (T @ np.concatenate([pose1_t, np.ones(1)]))[:3]

    new_R = T[:3, :3] @ Rotation.from_quat(gripper_pose_1[3:]).as_matrix()

    result = {
        "gripper_pose": np.concatenate([new_t, Rotation.from_matrix(new_R).as_quat()]),
        "gripper_frame": [
            new_t,
            new_R[0] * origin_frame_scale + new_t,
            new_R[1] * origin_frame_scale + new_t,
            new_R[2] * origin_frame_scale + new_t,
        ],
    }
    return result


def get_applicable_frame_idxes(kps, win_size=10):
    frames = []
    prev_kp = 0
    
    for k in kps:
        if k - win_size > prev_kp:
            frames += list(range(prev_kp, k - win_size))
        else:
            frames.append(prev_kp)
        prev_kp = k
    
    frames.append(prev_kp)
    return frames


if __name__ == "__main__":
    from PIL import Image
    from utils.structure import BASE_RLBENCH_TASKS
    root = "/scratch/xz653/code/RVT/data/train"
    
    print(get_applicable_frame_idxes([43, 80, 85, 100]))

    # if False:
    #     for task in tqdm(BASE_RLBENCH_TASKS):
    #         print(f'task - {task}')
    #         episode_path = osp.join(root, task, 'all_variations/episodes/episode0')
    #         mask =  np.array(Image.open(osp.join(episode_path, f"front_mask", '1.png'))).reshape(-1, 3)[:, 0]
    #         print(f'\t before: {set(np.unique(mask).tolist())}')
    #         mask = clean_mask(mask)
    #         mask_inds = np.unique(mask).tolist()
    #         for m in mask_inds:
    #             if len(str(m)) < 3 and m not in [0, 1]:
    #                 print(m)
    #         print(f'\t after: {set(mask_inds)}')

