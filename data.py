from typing import List
import torch
from PIL import Image
from utils.structure import BASE_RLBENCH_TASKS, load_json, load_pkl, dump_pkl, LOW_DIM_PICKLE, KEYPOINT_JSON, \
    DESC_PICKLE, VARIATION_NUMBER_PICKLE, dump_json
import os
from tqdm import tqdm
import os.path as osp
from collections import defaultdict
from utils.icp import *
from utils.vis import *
from rlbench.backend.utils import image_to_float_array
from pyrep.objects import VisionSensor
from rlbench.backend.const import DEPTH_SCALE
import heuristics as heu
from numpy.random import uniform 
from torch.utils.data import Dataset
from copy import copy, deepcopy
import fpsample
import open3d.ml.torch as ml3d
from utils.match import mutual_neighbor_correspondence
from geometry_lib import order_preserved_unique_np
cat = lambda *args, **kwargs: np.concatenate(args, **kwargs)


def merge_object_ids(object_ids):
    if hasattr(object_ids, 'misc'):
        object_ids = object_ids.misc['object_ids']
    objects = {}
    for k, v in object_ids.items():
        if v is not None:
            objects.update(v)
    return objects


def mask_post_process(mask, object_ids, return_dict=False):
    id2name = merge_object_ids(object_ids)
    name2id = {v:k for k,v in id2name.items()}

    keep_names = []
    exclude_robot_keywords = ['Panda']
    exclude_bg_keywords = ['Floor', 'Wall', 'drawer_legs', 'diningTable', 'topPlate',
                        'spawn_boundary', 'workspace']

    remapping = {
        'Dustpan_3': 'dustpan_tall',
        'Dustpan_5': 'dustpan_tall',
        **{f'dirt{i}':'dirt0' for i in range(1, 5)},
        'Dustpan_4': 'dustpan_short',
        'Dustpan_6': 'dustpan_short',
        'dollar_stack_boundary': 'safe_body',
    }

    new_mapping = {}
    mask = mask.copy()
    for mid in np.unique(mask):
        name = id2name[mid]
        if name in keep_names:
            new_mapping[mid] = name
            continue
        elif name in remapping:
            target_name = remapping[name]
            target_id = name2id[target_name]
            new_mapping[target_id] = target_name
            mask[mask == mid] = target_id
        elif any(kw in name for kw in exclude_bg_keywords):
            mask[mask == mid] = 0
            new_mapping[0] = 'background'
        elif any(kw in name for kw in exclude_robot_keywords):
            mask[mask == mid] = 1
            new_mapping[1] = 'robot'
        else:
            new_mapping[mid] = name
    if return_dict:
        return mask, new_mapping
    else:
        return mask

def query_next_kf(i, kps):
    for k in kps:
        if k > i:
            return k
    return i



def retreive_full_observation(essential_obs, episode_path, i, load_mask=False, skip_rgb=False):
    CAMERA_FRONT = 'front'
    CAMERA_LS = 'left_shoulder'
    CAMERA_RS = 'right_shoulder'
    CAMERA_WRIST = 'wrist'
    CAMERAS = [CAMERA_FRONT, CAMERA_LS, CAMERA_RS, CAMERA_WRIST]

    IMAGE_RGB = 'rgb'
    IMAGE_DEPTH = 'depth'
    IMAGE_FORMAT  = '%d.png'

    obs = {}

    if load_mask:
        for c in CAMERAS:
            obs[f"{c}_mask"] = np.array(
                Image.open(osp.join(episode_path, f"{c}_mask", IMAGE_FORMAT % i))
            )

    if not skip_rgb:
        obs['front_rgb'] =  np.array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs['left_shoulder_rgb'] = np.array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs['right_shoulder_rgb'] = np.array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs['wrist_rgb'] = np.array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_RGB), IMAGE_FORMAT % i)))

    obs['front_depth'] = image_to_float_array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    near = essential_obs.misc['%s_camera_near' % (CAMERA_FRONT)]
    far = essential_obs.misc['%s_camera_far' % (CAMERA_FRONT)]
    obs['front_depth'] = near + obs['front_depth'] * (far - near)

    obs['left_shoulder_depth'] = image_to_float_array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_LS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    near = essential_obs.misc['%s_camera_near' % (CAMERA_LS)]
    far = essential_obs.misc['%s_camera_far' % (CAMERA_LS)]
    obs['left_shoulder_depth'] = near + obs['left_shoulder_depth'] * (far - near)

    obs['right_shoulder_depth'] = image_to_float_array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_RS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    near = essential_obs.misc['%s_camera_near' % (CAMERA_RS)]
    far = essential_obs.misc['%s_camera_far' % (CAMERA_RS)]
    obs['right_shoulder_depth'] = near + obs['right_shoulder_depth'] * (far - near)

    obs['wrist_depth'] = image_to_float_array(Image.open(osp.join(episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
    near = essential_obs.misc['%s_camera_near' % (CAMERA_WRIST)]
    far = essential_obs.misc['%s_camera_far' % (CAMERA_WRIST)]
    obs['wrist_depth'] = near + obs['wrist_depth'] * (far - near)
    

    obs['front_point_cloud'] = VisionSensor.pointcloud_from_depth_and_camera_params(obs['front_depth'], 
                                                                                    essential_obs.misc['front_camera_extrinsics'],
                                                                                    essential_obs.misc['front_camera_intrinsics'])
    obs['left_shoulder_point_cloud'] = VisionSensor.pointcloud_from_depth_and_camera_params(obs['left_shoulder_depth'], 
                                                                                            essential_obs.misc['left_shoulder_camera_extrinsics'],
                                                                                            essential_obs.misc['left_shoulder_camera_intrinsics'])
    obs['right_shoulder_point_cloud'] = VisionSensor.pointcloud_from_depth_and_camera_params(obs['right_shoulder_depth'], 
                                                                                            essential_obs.misc['right_shoulder_camera_extrinsics'],
                                                                                            essential_obs.misc['right_shoulder_camera_intrinsics'])
    obs['wrist_point_cloud'] = VisionSensor.pointcloud_from_depth_and_camera_params(obs['wrist_depth'], 
                                                                                    essential_obs.misc['wrist_camera_extrinsics'],
                                                                                    essential_obs.misc['wrist_camera_intrinsics'])
    return obs


PCD, RGB, MASK, CLEAN_MASK = 0, 1, 2, 3
scene_bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]

def get_fg_from_mask(mask):
    return  ~((mask == 0) | (mask == 1))

def keep_valid_pcd(pc, other, scene_bounds=scene_bounds, return_indexes=False):
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
    if return_indexes:
        return ~inv_pnt
    else:
        return pc[~inv_pnt], tuple([x[~inv_pnt] for x in other])

def assemble_point_cloud(m, scene_bounds=scene_bounds, cameras=['front', 'left_shoulder', 'right_shoulder', 'wrist'], has_rgb=False, return_valid_indexes=False):
    pcd = []
    rgb = []
    mask = []
    for c in cameras:
        pcd.append(m[f"{c}_point_cloud"].reshape(-1, 3))
        if has_rgb:
            rgb.append(m[f"{c}_rgb"].reshape(-1, 3))
        mask.append(m[f"{c}_mask"].reshape(-1, 3))
    pcd = np.concatenate(pcd)
    if has_rgb:
        rgb = np.concatenate(rgb)
    else:
        rgb = np.zeros_like(pcd)
    mask = np.concatenate(mask)
    if return_valid_indexes:
        indexes = keep_valid_pcd(pcd, (rgb, mask), scene_bounds, return_indexes=True)
    if has_rgb:
        pcd, (rgb, mask) = keep_valid_pcd(pcd, (rgb, mask), scene_bounds)
    else:
        pcd, (mask, ) = keep_valid_pcd(pcd, (mask, ), scene_bounds)
    mask = mask[:, 0] + mask[:, 1] * 256 + mask[:, 2] * 256 * 256
    if return_valid_indexes:
        return pcd, rgb, mask, indexes
    else:
        return pcd, rgb, mask
    

def keypoint_discovery(demo, stopping_delta: float=0.1) -> List[int]:
    def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
        next_is_not_final = i != (len(demo) - 2)
        gripper_state_no_change = (
                i < (len(demo) - 2) and
                (obs.gripper_open == demo[i + 1].gripper_open and
                obs.gripper_open == demo[i - 1].gripper_open and
                demo[i - 2].gripper_open == demo[i - 1].gripper_open))
        small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
        stopped = (stopped_buffer <= 0 and small_delta and
                next_is_not_final and gripper_state_no_change)
        return stopped

    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or
                        last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    return episode_keypoints


PCD, RGB, MASK = 0, 1, 2


def compute_l1_dist_mat(p1, p2):
    return np.sum(np.abs(p1[:, None, :] - p2[None, :, :]), axis=-1)


class RLBenchDataset:
    def __init__(self, tasks=BASE_RLBENCH_TASKS, path='./datasets/base_training_set_raw', 
                cache_to='', cache_mode='read',
                max_episode_num=100, split='auto', grid_size=0.005,
                load_object_level_instruction=True,
                min_max_pts_per_obj=5000, device='cpu', use_lzma=True, 
                color_only_instructions=True, extend_key_objects=True):
        assert min_max_pts_per_obj is not None
        
        if split == 'auto':
            if osp.exists(osp.join(path, 'train')):
                split = 'train'
            else:
                split = ''
        if tasks == 'auto':       
            root = osp.join(path, split)
            tasks = []
            for t in os.listdir(root):
                if osp.isdir(osp.join(root, t)):
                    tasks.append(t)
            print(f'Tasks: {tasks}')
        if isinstance(tasks, str): tasks = [tasks,]
        self.tasks = tasks
        self.clip_device = device
        self.use_lzma = use_lzma
        self.extend_key_objects = extend_key_objects
        self.color_only_instructions = color_only_instructions
        self.split = split
        self.data_path = osp.join(path, split, '{}/all_variations/episodes/episode{}/')
        self.meta_path = osp.join(path, 'keyobjects/{}/{}/{}.json')
        self._episodes = defaultdict(list)
        self.grid_size = grid_size
        self.min_max_pts_per_obj = min_max_pts_per_obj
        self.load_object_level_instruction = load_object_level_instruction
        self._kfs_cache = {}
        self._desc_cache = {}
        self._ko_cache = {}
        self._debug_log = defaultdict(list)
        self.cache_folder = cache_to
        self.cache_mode = cache_mode
        
        for t in tasks:
            for e in range(max_episode_num):
                if osp.exists(self.data_path.format(t, e)): self._episodes[t].append(e)
    
    def instruction_classes(self):
        return heu.all_instructions(self.color_only_instructions)
    
    def get_episodes(self, task):
        return sorted(self._episodes.get(task, []))

    def get_kfs(self, task, e, exclude_last=True):
        if (task, e, exclude_last) in self._kfs_cache:
            return self._kfs_cache[(task, e, exclude_last)]

        keypoint_json = self.data_path.format(task, e) + KEYPOINT_JSON
        if not osp.exists(keypoint_json):
            obs = load_pkl(osp.join(self.data_path.format(task, e), LOW_DIM_PICKLE))
            kps = keypoint_discovery(obs)
            dump_json(osp.join(self.data_path.format(task, e), KEYPOINT_JSON), kps)
        kfs = [0] + load_json(keypoint_json)
        if exclude_last: kfs = kfs[:-1]
        if 1 in kfs: kfs.remove(1) # this could happens when robot simulator glitches at the beginning
        self._kfs_cache[(task, e, exclude_last)] = kfs
        return kfs
    
    def get_desc_and_vn(self, task, e):
        if (task, e) in self._desc_cache:
            return self._desc_cache[(task, e)]
        else:
            p = self.data_path.format(task, e)
            desc = load_pkl(osp.join(p, DESC_PICKLE))[0]
            vn = load_pkl(osp.join(p, VARIATION_NUMBER_PICKLE))
            self._desc_cache[(task, e)] = desc, vn
        return desc, vn

    def get_ko(self, task, e, kf):
        if (task, e, kf) in self._ko_cache:
            return self._ko_cache[(task, e, kf)]
        else:
            key_id = {int(v['segment']): (v['most_invariant_object'], v['most_invariant_object_name'])
                for v in load_json(self.meta_path.format(task, e, 'key_objects'))}[kf]
            self._ko_cache[(task, e, kf)] = key_id
            return key_id

    def load_instructions(self, result, training=True):
        task, desc = result['task'], result['desc']
        instructions = heu.parse_instructions(task, desc, color_only=self.color_only_instructions)

        result['instruction_class_names'] = [item[0] for item in instructions]
        for _, pos in instructions:
            assert pos >= 0
        result['instruction_classes'] = [heu.all_instructions(color_only=self.color_only_instructions).index(c) 
                                        for c in result['instruction_class_names']]
        result['instruction_positions'] = [item[1] for item in instructions]

        if training:
            e = result['e']
            object_ids = order_preserved_unique_np(result['mask'])
            object_names = [result['id2names'][oid] for oid in object_ids]
            # print(object_names)

            target_distractors_json = self.meta_path.format(task, e, 'target_distractors')
            if osp.exists(target_distractors_json):
                _ = load_json(target_distractors_json)['id']
                targets = [ result['id2names'][_['target']], ]
            else:
                # targets = 'jar0'
                targets = ['tap_left_visual']

            result['object_ids'] = object_ids
            result['object_names'] = object_names
            result['object_instructions'] = heu.assign_instruction_class_to_object(object_names, task, desc, targets=targets, 
                                                                                color_only=self.color_only_instructions)
        return result
    
    
    def find_grasped_object(self, pcd, clean_pcd, mask, clean_mask, obs, task):
        name2ids = {v:k for k,v in merge_object_ids(obs).items()}
        names = ['Panda_leftfinger_visual', 'Panda_rightfinger_visual']
        if obs.gripper_open: return -1
        fingers = [pcd[mask == name2ids[n]] for n in names if n in name2ids]
        if len(fingers) == 0:
            fingers = [pcd[mask == name2ids['Panda_gripper_visual']]]
        if task in ['push_buttons', 'slide_block_to_color_target']:
            return -1 # the gripper is tightly closed for these two tasks
        centers = [n.mean(axis=0) for n in fingers]
        dists, mask_ids = [], []
        for mid in np.unique(clean_mask):
            if mid in (0, 1): continue
            obj_center = clean_pcd[clean_mask == mid].mean(axis=0)
            dist = np.mean([np.linalg.norm(obj_center - c) for c in centers])
            dists.append(dist)
            mask_ids.append(mid)
        grasp_obj_id = mask_ids[np.argmin(dists)]
        return grasp_obj_id


    def get_next_kf(self, task, e, kf):
        return query_next_kf(kf, self.get_kfs(task, e, exclude_last=False))

    
    def sample(self, task, **kwargs):
        e = random.choice(self.get_episodes(task))
        kf = random.choice(self.get_kfs(task, e))
        return self.get(task, e, kf, **kwargs)
    
    
    def get_point_cloud_image(self, task, e, kf, camera='front'):
        data_path = self.data_path.format(task, e)
        rgb = np.array(Image.open(osp.join(data_path, f'{camera}_rgb',  f'{kf}.png')))
        depth =  image_to_float_array(Image.open(osp.join(data_path, f'{camera}_depth',  f'{kf}.png')), DEPTH_SCALE)
        
        obs = load_pkl(data_path + LOW_DIM_PICKLE)
        near = obs[kf].misc[f'{camera}_camera_near']
        far = obs[kf].misc[f'{camera}_camera_far']
        depth = near + depth * (far - near)
        pcd = VisionSensor.pointcloud_from_depth_and_camera_params(depth, obs[kf].misc[f'{camera}_camera_extrinsics'], obs[kf].misc[f'{camera}_camera_intrinsics'])
        return {'pcd': pcd, 'rgb': rgb}
    
    
    def get(self, task, e, kf, non_key_frame=False, skip_grasp=False, training=True, full=False):
        if 'read' == self.cache_mode and self.cache_folder:
            fname = osp.join(self.cache_folder, f'{task}-{e}-{kf}.pkl')
            result = load_pkl(fname, lzma=self.use_lzma)
            if self.load_object_level_instruction: result = self.load_instructions(result)
            if self.extend_key_objects: 
                result['key_ids'] = heu.extend_key_objects(result)
                result['key_names'] = [result['id2names'][k] for k in result['key_ids'] if k != -1]
                return result

        data_path = self.data_path.format(task, e)
        desc, vn = self.get_desc_and_vn(task, e)
        obs = load_pkl(data_path + LOW_DIM_PICKLE)
        if non_key_frame:
            next_kf = kf
            key_id = -1
        else:
            next_kf = query_next_kf(kf, self.get_kfs(task, e, exclude_last=False))
            try:
                key_id = {int(v['segment']): v['most_invariant_object'] 
                        for v in load_json(self.meta_path.format(task, e, 'key_objects'))}[kf]
            except (KeyError, FileNotFoundError):
                # assert self.get_kfs(task, e, exclude_last=False)[-1] == next_kf
                key_id = -1

        m = retreive_full_observation(obs[kf], data_path, kf, load_mask=True, skip_rgb=False)
        m = assemble_point_cloud(m, has_rgb=True)
        clean_mask, id2names = mask_post_process(m[MASK], obs[kf], return_dict=True)
        curr_pose, next_pose = obs[kf].gripper_pose, obs[next_kf].gripper_pose
        name2ids = {v: k for k, v in id2names.items()}

        if full:
            return {
                'pcd': m[PCD],
                'rgb': m[RGB],
                'mask': clean_mask,
                'id2names': id2names
            }
        
        fg_indexes = get_fg_from_mask(clean_mask)
        pcd, rgb, mask = m[PCD][fg_indexes], m[RGB][fg_indexes], clean_mask[fg_indexes]
        grasp_id = -1 if skip_grasp else self.find_grasped_object(m[PCD], pcd, m[MASK], mask, obs[kf], task)
        mask, grasp_id, key_id = self.merge_masks_for_certain_tasks(task, mask, name2ids, grasp_id, key_id)

        rgb = rgb.astype('float') / 127.5 - 1
        o3_pcd = to_o3d_pcd(pcd)
        o3_pcd.estimate_normals()
        normal = np.asarray(o3_pcd.normals)
        pcd, rgb, normal, mask = self.fps_subsample(*self.voxel_subsample(pcd, rgb, normal, mask))
        
        frame0 = pose7_to_frame(obs[0].gripper_pose)
        X_02t = pose7_to_X(curr_pose) @ inv(pose7_to_X(obs[0].gripper_pose))
        X_t2tp1 = pose7_to_X(next_pose) @ inv(pose7_to_X(curr_pose))
        frame_t = h_transform(X_02t, frame0)
        frame_tp1 = h_transform(X_t2tp1, frame_t)
        X_to_robot_frame = Rt_2_X(*arun(frame_t, X_to_frame(np.eye(4))))
        
        result = {
            'pcd': pcd,
            'rgb': rgb,
            'normal': normal,
            'mask': mask,
            'named_mask': to_named_masks(mask, id2names),

            'task': task,
            'desc': desc,
            'variation': vn,
            'e': e,
            'kf_t': kf,
            'kf_t+1': next_kf,

            'id2names': id2names,
            'name2ids': name2ids,
            'grasp_id': grasp_id,
            'key_id': key_id,

            'grasp_name': id2names.get(grasp_id, ''),
            'key_name': id2names.get(key_id, ''),

            'robot_pcd_0': frame0,
            'robot_pcd_t': frame_t,
            'robot_pcd_t+1': frame_tp1,
            'X_t2t+1': X_t2tp1,
            'X_02t': X_02t,
            'X_to_robot_frame': X_to_robot_frame,
            # arun(frame_t, frame_t+1) -> get X_t2t+1
            # X_to_pose(X_t2t+1 @ pose_to_X(curr_pose)) -> next_pose

            'pose_t': curr_pose,
            'open_t': obs[kf].gripper_open,
            'ignore_col_t': obs[kf].ignore_collisions,
            
            'pose_t+1':next_pose,
            'open_t+1': obs[next_kf].gripper_open,
            'ignore_col_t+1': obs[next_kf].ignore_collisions,
        }
        if 'write' in self.cache_mode:
            fname = osp.join(self.cache_folder, f'{task}-{e}-{kf}.pkl')
            dump_pkl(fname, result, lzma=self.use_lzma)

        if self.load_object_level_instruction:
            result = self.load_instructions(result, training)

        if self.extend_key_objects:
            result['key_ids'] = heu.extend_key_objects(result)
            result['key_names'] = [result['id2names'][k] for k in result['key_ids'] if k != -1]
        return result

    def size(self, episodes=None):
        total = 0
        for t in self.tasks:
            for e in episodes or self.get_episodes(t):
                for kf in self.get_kfs(t, e, exclude_last=False):
                    total += 1
        return total
    
    def prepare_obs(self, obs, pose0):
        data = {}
        object_ids = obs.pop('object_ids')
        for k, v in obs.items():
            if k in ['low_dim_state', 'task', 'desc', 'lang_goal_tokens', 'ignore_collisions', 'object_ids', 'gripper_pose', 'gripper_open', 'color_remap'] or 'trinsics' in k:
                continue
            data[k] = obs[k].reshape(3, -1).transpose(1, 0)
            if 'mask' in k:
                data[k] = (data[k] * 255).astype(int)
        
        m = assemble_point_cloud(data, has_rgb=True)
        clean_mask, id2names = mask_post_process(m[MASK], object_ids, return_dict=True)
        full_id2names = merge_object_ids(object_ids)
        name2ids = {v:k for k,v in id2names.items()}
        fg_indexes = get_fg_from_mask(clean_mask)
        pcd, rgb, mask = m[PCD][fg_indexes], m[RGB][fg_indexes], clean_mask[fg_indexes]

        rgb = rgb.astype('float') / 127.5 - 1
        o3_pcd = to_o3d_pcd(pcd)
        o3_pcd.estimate_normals()
        normal = np.asarray(o3_pcd.normals)
        pcd, rgb, normal, mask = self.fps_subsample(*self.voxel_subsample(pcd, rgb, normal, mask))

        frame0 = pose7_to_frame(pose0)
        X_02t = pose7_to_X(obs['gripper_pose']) @ inv(pose7_to_X(pose0))
        frame_t = h_transform(X_02t, frame0)
        X_to_robot_frame = Rt_2_X(*arun(frame_t, X_to_frame(np.eye(4))))

        result = {
            'pcd': pcd,
            'rgb': rgb,
            'normal': normal,
            'mask': mask,
            'named_mask': to_named_masks(mask, id2names),

            'task': obs['task'],
            'desc': obs['desc'],

            'id2names': id2names,
            'name2ids': name2ids,

            'key_id': -1,

            'robot_pcd_0': frame0,
            'robot_pcd_t': frame_t,

            'X_to_robot_frame': X_to_robot_frame,

            'pose_t': obs['gripper_pose'],
            'open_t': obs['gripper_open'],
            'ignore_col_t': obs['ignore_collisions'],
            
        }
        
        if self.load_object_level_instruction:
            result = self.load_instructions(result, training=False)
        return result
        
    
    def merge_masks_for_certain_tasks(self, task, mask, name2ids, go=-1, ko=-1):
        mask_mapping = {}
        if task == 'push_buttons':
            for i in range(3): mask_mapping[f'target_button_wrap{i}'] = f'push_buttons_target{i}'
        elif task == 'light_bulb_in':
            mask_mapping = {
                'light_bulb0': 'bulb0',
                'light_bulb1': 'bulb1'
            }
        elif task == 'beat_the_buzz':
            mask_mapping['wand_visual_sub'] = 'wand_visual'
        elif task == 'change_channel':
            mask_mapping.update({'target_button_wrap0': 'power_visual', 'target_button_wrap1': 'plus_visual'})
        elif task == 'hit_ball_with_queue':
            for i in range(2): mask_mapping[f'hit_ball_with_queue_pocket{i}'] = 'hit_ball_with_queue_pocket'
        elif task == 'hockey':
            mask_mapping['hockey_goal_visual'] = 'hockey_goal'
        elif task == 'open_window':
            mask_mapping['left_frame_visual'] = 'window_left'
        elif task == 'insert_usb_in_computer':
            mask_mapping.update({'usb_visual1': 'usb_visual0', 
                                 **{'Plane'+a: 'computer_visual' for a in ['', '0', '1', '2']}})      
        elif task == 'place_hanger_on_rack':
            mask_mapping['hanger_holder_visual'] = 'hanger_holder'            
        elif task == 'press_switch':
            mask_mapping.update({f'Shape_sub{a}': 'Shape_sub' for a in range(2)})
        elif task == 'put_books_at_shelf_location':
            for a in range(3): mask_mapping.update({f'book{a}_{b}': f'book{a}_front' for b in ['bottom','front','side','side_page','visual']})
        elif task == 'put_knife_on_chopping_board':
            mask_mapping['knife_block_visual'] = 'knife_visual'
        elif task == 'put_rubbish_in_color_bin':
            mask_mapping['color_bulb'] = 'light_bulb'
        elif task == 'screw_nail':
            mask_mapping['screw_driver_visual'] = 'screwdriver2'
        elif task in ['setup_checkers', 'setup_chess']:
            mask_mapping.update({f'squares_{a}': 'chess_board_base_visual' for a in ['black', 'white']})
        elif task == 'straighten_rope':
            mask_mapping.update({'tail': 'head', **{f'Cylinder{a}': 'head' for a in range(1, 17)}})
        elif task == 'turn_oven_on':
            mask_mapping['oven_stove_top0'] = 'oven_frame'
        elif task == 'wipe_desk':
            mask_mapping.update({f'Cuboid{a}':'Cuboid1' for a in range(100)})
        
        mask_mapping = {name2ids[k]: name2ids[v] for k, v in mask_mapping.items() if k in name2ids}
        mask = np.array([mask_mapping.get(int(v), int(v)) for v in mask])
        go = mask_mapping.get(go, go)
        ko = mask_mapping.get(ko, ko)
        return mask, go, ko 

    def voxel_subsample(self, pcd, rgb, normal, mask):
        if self.grid_size > 0:
            pooled_coord, pooled_feat = ml3d.ops.voxel_pooling(torch.from_numpy(pcd), torch.from_numpy(
                np.concatenate([rgb, normal, mask.astype('float')[:, None]], axis=1)).float(),
                                self.grid_size, position_fn='center', feature_fn='nearest_neighbor')
            pcd = pooled_coord.numpy()
            rgb = pooled_feat[:, :3].numpy()
            normal = pooled_feat[:, 3:6].numpy()
            mask = pooled_feat[:, -1].int().numpy()
        return pcd, rgb, normal, mask
    
    def fps_subsample(self, pcd, rgb, normal, mask):
        if self.min_max_pts_per_obj is not None:
            if isinstance(self.min_max_pts_per_obj, (tuple, list)):
                min_n, max_n = self.min_max_pts_per_obj 
                new_pcd, new_rgb, new_normal, new_mask = [], [], [], []
                for obj_id, cnt in zip(*np.unique(mask, return_counts=True)):
                    indices = (mask == obj_id).nonzero()[0].flatten()
                    if cnt > max_n:
                        idxs = fpsample.fps_sampling(pcd[indices], max_n)
                        indices = indices[idxs]
                    if cnt < min_n:
                        indices = cat(indices, np.random.choice(indices, size=min_n - cnt))
                    new_pcd.append(pcd[indices])
                    new_rgb.append(rgb[indices])
                    new_normal.append(normal[indices])
                    new_mask.append(mask[indices])
                pcd, rgb, normal, mask = cat(*new_pcd), cat(*new_rgb), cat(*new_normal), cat(*new_mask)
            else:
                if len(pcd) > self.min_max_pts_per_obj: 
                    idxs = fpsample.fps_sampling(pcd, self.min_max_pts_per_obj)
                    pcd, rgb, normal, mask = pcd[idxs], rgb[idxs], normal[idxs], mask[idxs]

        return pcd, rgb, normal, mask
    

class RLBenchTransitionPairDataset(Dataset):
    def __init__(self, dset: RLBenchDataset, size=5000, 
                full_iteration=False, cache_to="",
                aug_xyz=[0.125, 0.125, 0.0], corr_search_radius=0.05, correspondence=True,
                align_twice=False, include_T=False, noisy_mask=0,
                aug_rpy=[0, 0, 15.0], use_aug=False, match_vn=False):
        super().__init__()
        self.dset = dset
        self._size = size
        self.noisy_mask = noisy_mask
        self.align_twice = align_twice
        self.include_T = include_T
        self.corr_search_radius = corr_search_radius
        self.correspondence = correspondence
        self.full_iteration = full_iteration
        self.use_aug = use_aug
        self.aug_xyz, self.aug_rpy = np.array(aug_xyz), np.array(aug_rpy)
        self.pairs = load_pkl(cache_to)
        if full_iteration: self._to_flat_pairs()
    

    def add_noise(self, mask, ignore_value, candidates=None):
        mask = mask.copy()
        background = mask == ignore_value
        if candidates is None:
            candidates = list(set(mask.tolist()))
            if ignore_value in candidates:
                candidates.remove(ignore_value)
        noise = np.random.choice(candidates, size=mask.shape)

        num = min(int(background.sum() * self.noisy_mask), np.prod(mask.shape) - background.sum())
        pick = np.full(len(mask), False)
        pick[:num] = True
        np.random.shuffle(pick)

        pick *= background
        mask[pick] = noise[pick]
        return mask 


    def summarize(self, logger=print, tasks=None, skip_full_report=['reach_and_drag', 'push_buttons', 
                    'stack_cups', 'light_bulb_in', 'stack_blocks', 'insert_onto_square_peg', 'close_jar']):
        assert not self.full_iteration
        tasks = tasks or list(self.pairs.keys())
        for t in tasks:
            episodes = self.pairs[t]
            logger(f'> {t} - {len(episodes)} episodes')
            all_vns = {}
            e2vn = {}
            for e in range(100):
                desc, vn = self.dset.get_desc_and_vn(t, e)
                all_vns[vn] = desc
                e2vn[e] = vn
            has_vns = {self.dset.get_desc_and_vn(t, e)[1] for e in episodes}
            logger(f'\t variations: {len(has_vns)} kept from {len(all_vns)} (total)')
            if len(has_vns) < len(all_vns):
                for vn, desc in all_vns.items():
                    if vn not in has_vns:
                        logger(f'\t\t missing: {vn} - {desc}')
            full_matched = {vn: 0 for vn in all_vns}
            for e, kfs in episodes.items():
                if set(kfs) == set(self.dset.get_kfs(t, e)):
                    full_matched[e2vn[e]] += 1
            total = sum(full_matched.values())
            logger(f'\t # of full episodes {total}')
            if t not in skip_full_report:
                for vn, num in full_matched.items():
                    logger(f'\t\t - vn:{vn} - {all_vns[vn]} - {num}')

    
    def _to_flat_pairs(self):
        if not isinstance(self.pairs, list):
            all_pairs = []
            for t in self.pairs:
                for e in self.pairs[t]:
                    for kf in self.pairs[t][e]:
                        for ref_e, ref_kf in self.pairs[t][e][kf]:
                            all_pairs.append((t, e, kf, ref_e, ref_kf))
            self.pairs = all_pairs

    def _get_index(self, index, debug):
        pairs = self.pairs
        if debug:
            if isinstance(index, str):
                task = index
                e = random.choice(list(pairs[task].keys()))
                kf = random.choice(list(pairs[task][e].keys()))
                ref_e, ref_kf = random.choice(pairs[task][e][kf])
            else:
                if len(index) == 3:
                    task, e, kf = index
                    ref_e, ref_kf = random.choice(pairs[task][e][kf])
                else:
                    task, e, kf, ref_e, ref_kf = index
        else:
            if self.full_iteration:
                task, e, kf, ref_e, ref_kf = pairs[index]
            else:
                task = random.choice(self.dset.tasks)
                e = random.choice(list(pairs[task].keys()))
                kf = random.choice(list(pairs[task][e].keys()))                   
                ref_e, ref_kf = random.choice(pairs[task][e][kf])
        return  task, e, kf, ref_e, ref_kf 

    
    def __getitem__(self, index, debug=False, training=True):
        task, e, kf, ref_e, ref_kf = index = self._get_index(index, debug)
        src_t = self.dset.get(task, ref_e, ref_kf, training=training)
        tgt_t = self.dset.get(task, e, kf, training=training)
        src, tgt = src_t, tgt_t
        src_t1, tgt_t1 = self.dset.get(task, ref_e, src_t['kf_t+1'], training=training), self.dset.get(task, e, tgt_t['kf_t+1'], training=training)

        if self.use_aug:
            src_aug_t, tgt_aug_t = uniform(-1, 1, size=3) * self.aug_xyz, uniform(-1, 1, size=3) * self.aug_xyz
            src_aug_rpy, tgt_aug_rpy = uniform(-1, 1, size=3) * self.aug_rpy, uniform(-1, 1, size=3) * self.aug_rpy
            src_t, src_t1 = self.augment(src_t, src_aug_t, src_aug_rpy), self.augment(src_t1, src_aug_t, src_aug_rpy) 
            tgt_t, tgt_t1 = self.augment(tgt_t, tgt_aug_t, tgt_aug_rpy), self.augment(tgt_t1, tgt_aug_t, tgt_aug_rpy)
        

        for item in [src_t, tgt_t, src_t1, tgt_t1]:
            if item is None: continue
            if len(item['instruction_classes']) > 0:
                int2pos = dict(zip(item['instruction_classes'], item['instruction_positions']))
                obj2int = dict(zip(item['object_ids'], item['object_instructions']))
                item['instruction_mask'] =  np.array([obj2int.get(int(v), -1) for v in item['mask']], dtype=np.int64)
                item['position_mask'] = np.array([int2pos.get(int(v), -1) for v in item['instruction_mask']], dtype=np.int64)
                if self.noisy_mask > 0:
                    item['noisy_instruction_mask'] = self.add_noise(item['instruction_mask'], -1)
                    item['noisy_position_mask'] = np.array([int2pos.get(int(v), -1) for v in item['noisy_instruction_mask']], dtype=np.int64)
            else:
                item['position_mask'] = np.full([len(item['pcd']), ], fill_value=-1, dtype=np.int64)
                item['instruction_mask'] = np.full([len(item['pcd']), ], fill_value=-1, dtype=np.int64)
                if self.noisy_mask > 0:
                    item['noisy_position_mask'] = item['position_mask']
                    item['noisy_instruction_mask'] = item['instruction_mask'] 

        if 'key_ids' in src:
            src['key_mask'] = np.isin(src['mask'], src['key_ids'])
            src_t1['key_mask'] = np.isin(src_t1['mask'], src['key_ids'])
            tgt['key_mask'] = np.isin(tgt['mask'], tgt['key_ids'])
        else:
            src['key_mask'] = src['mask'] == src['key_id']
            tgt['key_mask'] = tgt['mask'] == tgt['key_id']
            src_t1['key_mask'] = src_t1['mask'] = src['key_id']
        
        if self.correspondence:
            src_to_tgt_X = Rt_2_X(*arun(src['robot_pcd_t+1'], tgt['robot_pcd_t+1']))
            src_ko_indices = src['key_mask'].nonzero()[0].flatten()
            tgt_ko_indices = tgt['key_mask'].nonzero()[0].flatten()

            if self.align_twice:
                icp_result = icp(src['pcd'][src_ko_indices], tgt['pcd'][tgt_ko_indices], init_X=src_to_tgt_X)
                src_to_tgt_X = icp_result.transformation
            corr = mutual_neighbor_correspondence(
                                h_transform(src_to_tgt_X, src['pcd'][src_ko_indices]),
                                tgt['pcd'][tgt_ko_indices], search_radius=self.corr_search_radius).T

            ko_corr = np.copy(corr)
            corr[:, 0] = src_ko_indices[corr[:, 0]]
            corr[:, 1] = tgt_ko_indices[corr[:, 1]] 
            correspondence = {'src_to_tgt_X': src_to_tgt_X, 'correspondence': corr, 'ko_correspondence': ko_corr}
        else:
            correspondence = None
        
        result = {
            'src': {'t': src_t, 't+1': src_t1},
            'tgt': {'t': tgt_t, 't+1': tgt_t1},
            'match': correspondence,
            'index': index
        }

        if self.include_T:
            ref_kf_last = self.dset.get_kfs(task, ref_e, exclude_last=False)[-1]
            src_T = self.dset.get(task, ref_e, ref_kf_last)

            if self.use_aug:
                src_T = self.augment(src_T, src_aug_t, src_aug_rpy)
            result['src']['T'] = src_T
        return result

        
    def augment(self, item, t, rpy):
        """
        warning! this augmentation will invalidate many transformation stored in the original payload,
        here we only adjust the X_to_robot_frame
        """
        if item is None: return None
        R = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
        X = Rt_2_X(R, t)
        for k in list(item.keys()):
            if 'pcd' in k: item[k] = h_transform(X, item[k]) 
            if 'pose' in k: item[k] = h_transform_pose(X, item[k])
        
        item['X_to_robot_frame'] = Rt_2_X(*arun(item['robot_pcd_t'], X_to_frame(np.eye(4))))
        return item

    
    def __len__(self):
        if self.full_iteration:
            return len(self.pairs)
        else:
            return self._size
    
    def size(self):
        total = 0
        for t in self.pairs:
            for e in self.pairs[t]:
                for kf in self.pairs[t][e]:
                    total += len(self.pairs[t][e][kf])
        return total

    def get(self, index, training=True):
        return  self.__getitem__(index, debug=True, training=training)


def to_torch(result):
    if isinstance(result, list):
        if len(result) == 0: return None
        if isinstance(result[0], np.ndarray):
            tensor = torch.from_numpy(cat(*result))
            dtype = result[0].dtype
            if 'float' in str(dtype):
                tensor = tensor.float()
            elif 'bool' in str(dtype):
                tensor = tensor.bool()
            else:
                tensor = tensor.long()
        else:
            tensor = torch.as_tensor(result)
            if isinstance(result[0], int): tensor = tensor.long()
            else: tensor = tensor.float(0)
        return tensor
    else:
        assert isinstance(result, dict)
        meta = None
        if 'meta' in result:
            meta = result.pop('meta')
        result = {k: to_torch(v) for k, v in result.items()}
        result = {k: v for k, v in result.items() if v is not None}
        if meta is not None: result['meta'] = meta
        return result


def to_device(data, device):
    if data is None: return data
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=True)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], torch.Tensor):
            return [a.to(device, non_blocking=True) for a in data]
        else:
            return data
    else:
        raise ValueError()


class RLBenchCollator:
    def __init__(self, use_segmap=True, training=True):
        self.use_segmap = use_segmap
        self.training = training

    def __call__(self, samples):
        item = { 'pcd': [], 'rgb': [], 'normal': [], 'batch_index': [], 
                'mask': [], 'key_mask': [], 'position_mask': [], 'instruction_mask': [],

                'noisy_key_mask': [], 'noisy_position_mask': [], 'noisy_instruction_mask': [],

                'robot_position': [], 'open': [], 'ignore_col': [], 'robot_position_t+1': [],
                'X_to_robot_frame': [], 

                'obj_batch_index': [], 'obj_level_offset': [], 
                'key_object': [], 

                'ko_pcd': [], 'ko_rgb': [], 'ko_normal': [], 'ko_batch_index': [], 
                'ctx_pcd': [], 'ctx_rgb': [], 'ctx_normal': [], 'ctx_batch_index': [] }

        instruction_item =  {
                                'object_instructions': [],
                                'instructions': [],
                                'instruction_positions': []     
                            }
        result = { 'src': { 't': deepcopy(item), 't+1': deepcopy(item), 'T': deepcopy(item)},
                'tgt': { 't': deepcopy(item), 't+1': deepcopy(item) },  
                'meta': { 'data_index': [], 'X': [], 'ko_correspondence': [], 'correspondence': [], 
                        'desc': {'src': {'t': [], 't+1': []}, 'tgt': {'t': [], 't+1': []}},
                        'id2names': {'src': {'t': [], 't+1': []}, 'tgt': {'t': [], 't+1': []}},
                        'instructions': {
                            'src': {'t': deepcopy(instruction_item), 't+1': deepcopy(instruction_item)},
                            'tgt': {'t': deepcopy(instruction_item), 't+1': deepcopy(instruction_item)},
                        }}}
        if not self.training and 'tgt' in samples[0] and 't+1' not in samples[0]['tgt']: 
            result['tgt'].pop('t+1')
            
        num_pts = {'src': {'t': 0, 't+1': 0}, 'tgt': {'t': 0, 't+1': 0}}
        num_objs = {'src': {'t': 0, 't+1': 0}, 'tgt': {'t': 0, 't+1': 0}}
        for sample_id, item in enumerate(samples):
            for k1 in ['src', 'tgt']:
                if k1 not in item:
                    continue 
                
                for k2 in ['t', 't+1', 'T']:
                    if k2 not in item[k1]: 
                        if k2 in result[k1]: result[k1].pop(k2)
                        continue
                    source = item[k1][k2]
                    target = result[k1][k2]

                    target['batch_index'].append(np.full([len(source['pcd'])], fill_value=sample_id))
                    num_pts[k1][k2] += len(source['pcd'])

                    if self.use_segmap:
                        obj_ids, obj_indexes = order_preserved_unique_np(source['mask'], return_inverse=True)
                        target['obj_batch_index'] += [obj_indexes + num_objs[k1][k2]]
                        num_objs[k1][k2] += len(obj_ids)
                        target['obj_level_offset'].append(num_objs[k1][k2])
                    else:
                        if 'instruction_mask' in source:
                            target['instruction_mask'].append(source['instruction_mask'])
                        if 'position_mask' in source:
                            target['position_mask'].append(source['position_mask'])
                        if 'noisy_position_mask' in source:
                            target['noisy_instruction_mask'].append(source['noisy_instruction_mask'])
                            target['noisy_position_mask'].append(source['noisy_position_mask'])
                        
                        if 'mask' in source:
                            _, obj_indexes = order_preserved_unique_np(source['mask'], return_inverse=True)
                            target['obj_batch_index'].append(obj_indexes)

                    target['pcd'].append(source['pcd'])
                    target['rgb'].append(source['rgb'])
                    if 'mask' in source:
                        target['mask'].append(source['mask'])
                    target['normal'].append(source['normal'])

                    target['open'].append(int(source['open_t']))
                    target['ignore_col'].append(int(source['ignore_col_t']))
                    target['robot_position'].append(source['robot_pcd_t'][None, ...])

                    target['X_to_robot_frame'].append(source['X_to_robot_frame'][None, ...])


                    if k2 == 't':
                        if source.get('robot_pcd_t+1', None) is not None:
                            target['robot_position_t+1'].append(source['robot_pcd_t+1'][None, ...])

                        if self.use_segmap:
                            if 'key_id' in source and source['key_id'] not in [-1, None]:
                                ko_indices = source['mask'] == source['key_id']
                                ctx_indices = source['mask'] != source['key_id']

                                for group_name, indices in [('ko_', ko_indices), ('ctx_', ctx_indices)]:
                                    target[group_name + 'pcd'].append(source['pcd'][indices])
                                    target[group_name + 'rgb'].append(source['rgb'][indices])
                                    target[group_name + 'normal'].append(source['normal'][indices])
                                    target[group_name + 'batch_index'] += [np.full([len(target[group_name + 'pcd'][-1])], fill_value=sample_id)]
                            
                            if source.get('key_id', -1) not in [-1, None]:
                                target['key_object'].append(int(source['key_id']))

                            if self.training:
                                result['meta']['instructions'][k1]['object_instructions'].append(source['object_instructions'])
                        else:
                            if 'key_mask' in source:
                                target['key_mask'].append(source['key_mask'])
                            if self.training and 'noisy_key_mask' in source:
                                target['noisy_key_mask'].append(source['noisy_key_mask'])
                    
                    if k1 == 'src' and k2 == 't+1':
                        if self.training and 'key_mask' in source:
                            target['key_mask'].append(source['key_mask'])

                    if k2 in ['t', 't+1']:
                        if 'desc' in source:
                            result['meta']['instructions'][k1][k2]['instructions'].append(source['instruction_classes'])
                            result['meta']['instructions'][k1][k2]['instruction_positions'].append(source['instruction_positions'])
                            result['meta']['desc'][k1][k2].append(source['desc'])
                            result['meta']['id2names'][k1][k2].append(source['id2names'])


            if 'match' in item and item['match'] is not None:
                result['meta']['ko_correspondence'].append(torch.from_numpy(item['match']['ko_correspondence']))
                result['meta']['correspondence'].append(torch.from_numpy(item['match']['correspondence']))
                result['meta']['X'].append(item['match']['src_to_tgt_X'])
            
            if self.training:
                result['meta']['data_index'].append(item['index'])
        return to_torch(result)
