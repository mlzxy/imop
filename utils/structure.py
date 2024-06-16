from typing import Any, List
import lzma as lzmalib
import os
import os.path as osp
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import pickle
import json
from abc import ABC, abstractmethod
import base64
from hashlib import blake2b


LOW_DIM_PICKLE = 'low_dim_obs.pkl'
KEYPOINT_JSON = "keypoints.json"
VARIATION_NUMBER_PICKLE = 'variation_number.pkl'
LANG_GOAL_EMB = "lang_emb.pkl"
DESC_PICKLE = "variation_descriptions.pkl"


BASE_RLBENCH_TASKS = [
    "put_item_in_drawer",
    "reach_and_drag",
    "turn_tap",
    "slide_block_to_color_target",
    "open_drawer",
    "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter",
    "put_money_in_safe",
    "push_buttons",
    "close_jar",
    "stack_blocks",
    "place_cups",
    "place_wine_at_rack_location",
    "light_bulb_in",
    "sweep_to_dustpan_of_size",
    "insert_onto_square_peg",
    "meat_off_grill",
    "stack_cups",
]

NOVEL_RLBENCH_TASKS = [
"basketball_in_hoop", 
"put_rubbish_in_bin",
"scoop_with_spatula", 
"place_hanger_on_rack", 
"hit_ball_with_queue", 
"block_pyramid", 
"take_lid_off_saucepan", 
"lamp_on", 
"phone_on_base", 
"open_box", 
"close_laptop_lid", 
"beat_the_buzz", 
"remove_cups", 
"play_jenga", 
"put_knife_on_chopping_board", 
"straighten_rope", 
"change_clock", 
"open_wine_bottle", 
"open_door",
"take_money_out_safe", 
"close_microwave",
"slide_cabinet_open_and_place_cups"
]



def load_pkl(fp, lzma=False):
    if lzma:
        with lzmalib.open(fp, 'rb') as f:
            return pickle.load(f)
    else:
        with open(fp, 'rb') as f:
            return pickle.load(f)
    
def as_list(x):
    if isinstance(x, list): 
        return x
    elif isinstance(x, tuple):
        return list(x)
    elif hasattr(x, 'tolist'):
        return x.tolist()
    else:
        return [x]
    

def load_json(fp):
    with open(fp, 'r') as f:
        return json.load(f)
    
    
def ensure_dir(d):
    if d: os.makedirs(d, exist_ok=True)

def dump_pkl(fp, obj, lzma=False):
    ensure_dir(osp.dirname(fp))
    if lzma:
        with lzmalib.open(fp, 'wb') as f:
            return pickle.dump(obj, f)
    else:
        with open(fp, 'wb') as f:
            return pickle.dump(obj, f)
    

def dump_json(fp, obj, **kwargs):
    ensure_dir(osp.dirname(fp))
    with open(fp, 'w') as f:
        return json.dump(obj, f, **kwargs)

@dataclass
class ActResult:
    action: Any
    observation_elements: dict = field(default_factory=dict)
    replay_elements: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

@dataclass
class ObservationElement:
    name: str
    shape: tuple
    type: Any


@dataclass
class Summary:
    name: str
    value: Any

@dataclass
class TextSummary(Summary): pass

ScalarSummary = TextSummary

@dataclass
class ImageSummary(Summary): pass

@dataclass
class VideoSummary: 
    name: str
    value: Any
    fps: int = 30


@dataclass
class Transition:
    observation: dict
    reward: float
    terminal: bool
    info: dict = field(default_factory=dict)
    summaries: List[Summary] = field(default_factory=list)

@dataclass
class FullTransition:
    observation: dict
    action: np.ndarray
    reward: float
    terminal: bool
    timeout: bool
    summaries: List[Summary] = field(default_factory=list)
    info: dict = field(default_factory=dict)


ROBOT_STATE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
                    'gripper_open', 'gripper_pose',
                    'gripper_joint_positions', 'gripper_touch_forces',
                    'task_low_dim_state', 'misc']




class Env(ABC):

    def __init__(self):
        self._active_task_id = 0
        self._eval_env = False

    @property
    def eval(self):
        return self._eval_env

    @eval.setter
    def eval(self, eval):
        self._eval_env = eval

    @property
    def active_task_id(self) -> int:
        return self._active_task_id

    @abstractmethod
    def launch(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> dict:
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> Transition:
        pass

    @property
    @abstractmethod
    def observation_elements(self) -> List[ObservationElement]:
        pass

    @property
    @abstractmethod
    def action_shape(self) -> tuple:
        pass

    @property
    @abstractmethod
    def env(self) -> Any:
        pass

    @property
    @abstractmethod
    def num_tasks(self) -> int:
        pass
    
    


class Agent(ABC):

    @abstractmethod
    def build(self, training: bool, device=None) -> None:
        pass

    @abstractmethod
    def update(self, step: int, replay_sample: dict, **kwargs) -> dict:
        pass

    @abstractmethod
    def act(self, step: int, observation: dict, **kwargs) -> ActResult:
        # returns dict of values that get put in the replay.
        # One of these must be 'action'.
        pass

    def reset(self) -> None:
        pass
    
    def reset_to_demo(self, i: int, variation_number: int=-1) -> None:
        pass



@dataclass
class DataElement:
    name: str
    shape: tuple
    type: Any
    is_observation: bool = False
    
    

def hash_object(*objs):
    h = blake2b(digest_size=20)
    h.update(pickle.dumps(objs))
    return h.hexdigest()

