import numpy as np
"""
put_item_in_drawer,
reach_and_drag,
turn_tap,
slide_block_to_color_target,
open_drawer,
put_groceries_in_cupboard,
place_shape_in_shape_sorter,
put_money_in_safe,
push_buttons,
close_jar,
stack_blocks,
place_cups,
place_wine_at_rack_location,
light_bulb_in,
sweep_to_dustpan_of_size,
insert_onto_square_peg,
meat_off_grill,
stack_cups
"""

SPATIAL_DIRECTIVES = ['top', 'middle', 'bottom', 'left', 'right']

def parse_spatial_directive(desc):
    for n in SPATIAL_DIRECTIVES:
        if n in desc:
            return n

def parse_number(desc):
    for i in range(1, 5):
        if str(i) in desc: return i


def object_shall_be_movable(task, desc, oname):
    exact = False
    if task == 'put_item_in_drawer':
        words = ['item', parse_spatial_directive(desc)]
    elif task == 'reach_and_drag':
        words = ['cube', 'stick']
    elif task in ['turn_tap', 'open_drawer']:
        words = [parse_spatial_directive(desc)]
    elif task == 'slide_block_to_color_target':
        words = ['block']
    elif task == 'put_groceries_in_cupboard':
        oname = oname.split("_")[0]
        return oname in desc and oname != 'cupboard'
    elif task == 'place_shape_in_shape_sorter':
        oname = oname.split("_")[0]
        return oname in desc and oname != 'shape'
    elif task == 'put_money_in_safe':
        words = ['dollar_stack']
    elif task == 'push_buttons': 
        return False
    elif task == 'close_jar':
        words = ['jar_lid0']
    elif task == 'stack_blocks':
        return [f'target{i}' for i in range(parse_number(desc))]
    elif task == 'place_cups':
        words = [f'mug_visual{i}' for i in range(parse_number(desc))] 
    elif task == 'place_wine_at_rack_location':
        words = ['wine_bottle']
    elif task == 'light_bulb_in':
        words = ['bulb1', 'bulb0']
        exact = True
    elif task == 'sweep_to_dustpan_of_size':
        words = ['broom_visual', 'dirt0'] 
    elif task == 'insert_onto_square_peg':
        words = ['square_ring']
    elif task == 'meat_off_grill':
        words = ['chicken', 'steak']
    elif task == 'stack_cups':
        words = ['cup']
    else:
        raise KeyError('unrecognized task: ' + task)
    if exact:
        return any([w == oname for w in words])
    else:
        return any([w in oname for w in words])


def number_of_movable_objects_at_once(task, desc):
    if task in ['reach_and_drag', 'sweep_to_dustpan_of_size']:
        return 2
    else:
        return 1
    

colors = dict([
        ('red', (1.0, 0.0, 0.0)),
        ('maroon', (0.5, 0.0, 0.0)),
        ('lime', (0.0, 1.0, 0.0)),
        ('green', (0.0, 0.5, 0.0)),
        ('blue', (0.0, 0.0, 1.0)),
        ('navy', (0.0, 0.0, 0.5)),
        ('yellow', (1.0, 1.0, 0.0)),
        ('cyan', (0.0, 1.0, 1.0)),
        ('magenta', (1.0, 0.0, 1.0)),
        ('silver', (0.75, 0.75, 0.75)),
        ('gray', (0.5, 0.5, 0.5)),
        ('orange', (1.0, 0.5, 0.0)),
        ('olive', (0.5, 0.5, 0.0)),
        ('purple', (0.5, 0.0, 0.5)),
        ('teal', (0, 0.5, 0.5)),
        ('azure', (0.0, 0.5, 1.0)),
        ('violet', (0.5, 0.0, 1.0)),
        ('rose', (1.0, 0.0, 0.5)),
        ('black', (0.0, 0.0, 0.0)),
        ('white', (1.0, 1.0, 1.0)),
        ('pink', (0.95, 0.075, 0.54))
    ])

COLOR_NAMES = list(colors.keys())

def find_color_directive(desc, all_colors=None):
    cs = []
    if all_colors is None: all_colors = colors.keys()
    for c in all_colors:
        c = f' {c} '
        if c in desc:
            cs.append((desc.index(c), c.strip()))
    cs = sorted(cs)
    return [c for _, c in cs]


GROCERY_NAMES = [
    'crackers',
    'chocolate jello',
    'strawberry jello',
    'soup',
    'tuna',
    'spam',
    'coffee',
    'mustard',
    'sugar',
] 

SHAPE_NAMES = ['cube', 'cylinder', 'triangular prism', 'star', 'moon']

COLOR_RVT_TASKS = [
    "reach_and_drag",
    "push_buttons",
    "close_jar",
    "stack_blocks",
    "light_bulb_in",
    "insert_onto_square_peg",
    "stack_cups",
    "block_pyramid",
    "slide_block_to_color_target"
]

def all_instructions(color_only=False):
    if color_only:
        return COLOR_NAMES
    else:
        ALL_INSTRUCTIONS = GROCERY_NAMES + SHAPE_NAMES + COLOR_NAMES + SPATIAL_DIRECTIVES
        return ALL_INSTRUCTIONS

def find_tag_indexes(desc, tags):
    indexes = []
    for t in tags:
        indexes.append(desc.index(t))
    return indexes

def parse_instructions(task, desc, color_only=False):
    tags = []
    if color_only:
        if task in COLOR_RVT_TASKS:
            tags = find_color_directive(desc, all_instructions(color_only))
    else:
        if task == 'place_shape_in_shape_sorter':
            for g in SHAPE_NAMES:
                if g in desc:
                    tags.append(g)
        elif task == 'put_groceries_in_cupboard':
            for g in GROCERY_NAMES:
                if g in desc:
                    tags.append(g)
        elif task in ['open_drawer', 'put_item_in_drawer', 'turn_tap']:
            tags = [parse_spatial_directive(desc), ] 
        else:
            tags = find_color_directive(desc)
    indexes = find_tag_indexes(desc, tags)
    return list(zip(tags, indexes))


def list_index(lst, v):
    if v in lst:
        return lst.index(v)
    else:
        return -1


def assign_instruction_class_to_object(object_names, task, desc, targets=None, color_only=False):
    instructions = parse_instructions(task, desc, color_only=color_only)
    if len(instructions) == 0: return [-1] * len(object_names)
    ALL_INSTRUCTIONS = all_instructions(color_only)
    if color_only and task not in COLOR_RVT_TASKS:
        return [-1] * len(object_names)

    if task == 'put_groceries_in_cupboard':
        grocery_names = [a.replace(' ', '_') + '_visual' for a in GROCERY_NAMES]
        indexes = [list_index(grocery_names, obj) for obj in object_names]
        class_indexes = [ALL_INSTRUCTIONS.index(GROCERY_NAMES[i]) if i != -1 else i for i in indexes]

    elif task == 'place_shape_in_shape_sorter':
        shape_names = [(a.replace(' ', '_') + '_visual') if a in ('star', 'moon') else a for a in SHAPE_NAMES]
        indexes = [list_index(shape_names, obj) for obj in object_names]
        class_indexes = [ALL_INSTRUCTIONS.index(SHAPE_NAMES[i]) if i != -1 else i for i in indexes]

    elif task == 'slide_block_to_color_target':
        class_indexes = []
        for o in object_names:
            ind = list_index(['target1', 'target2', 'target3', 'target4'], o)
            if ind != -1:
                ind = ALL_INSTRUCTIONS.index(['green', 'blue', 'pink', 'yellow'][ind])
            class_indexes.append(ind)

    elif task in ['open_drawer', 'put_item_in_drawer',]:
        class_indexes = []
        for o in object_names:
            ind = list_index(['drawer_top', 'drawer_middle', 'drawer_bottom'], o)
            if ind != -1:
                ind = ALL_INSTRUCTIONS.index(['top', 'middle', 'bottom'][ind])
            class_indexes.append(ind)

    elif task ==  'turn_tap':
        class_indexes = []
        for o in object_names:
            ind = list_index(['tap_right_visual', 'tap_left_visual'], o)
            if ind != -1:
                ind = ALL_INSTRUCTIONS.index(['right', 'left'][ind])
            class_indexes.append(ind)

    elif task == 'stack_cups':
        assert len(instructions) == 1
        target_color = instructions[0][0]
        class_indexes = [ALL_INSTRUCTIONS.index(target_color) if o == 'cup2_visual' else -1 for o in object_names]
        
    elif task == 'push_buttons':
        assert len(instructions) >= 1
        target_colors = [c for c, _ in instructions]
        target_names = [f'push_buttons_target{i}' for i in range(len(instructions))] 
        class_indexes = []
        for o in object_names:
            ind = list_index(target_names, o)
            if ind != -1:
                ind = ALL_INSTRUCTIONS.index(target_colors[ind])
            class_indexes.append(ind)

    elif task == 'stack_blocks':
        class_indexes = []
        assert len(instructions) == 1
        target_color = instructions[0][0]
        target_color_clsind = ALL_INSTRUCTIONS.index(target_color)
        for name in object_names:
            if 'target' in name and 'target_plane' not in name:
                class_indexes.append(target_color_clsind)
            else:
                class_indexes.append(-1)

    elif task == 'reach_and_drag':
        assert len(instructions) == 1
        target_color = instructions[0][0]
        target_color_clsind = ALL_INSTRUCTIONS.index(target_color)
        class_indexes = [target_color_clsind if 'target' in o else -1 for o in object_names]

    elif task in [ 'light_bulb_in', 'close_jar',  'insert_onto_square_peg']:
        assert len(instructions) == 1
        assert targets is not None
        target_color = instructions[0][0]
        target_color_clsind = ALL_INSTRUCTIONS.index(target_color)
        class_indexes = [target_color_clsind if o in targets else -1 for o in object_names]
        
    else:
        raise KeyError(task)
    
    return class_indexes



def extend_key_objects(item):
    task, desc = item['task'], item['desc']
    if item['key_id'] == -1: return [-1]
    result = [item['key_id'], ]
    def find_ids(names): return [item['name2ids'][n] for n in names]

    if task in [ "push_buttons", "meat_off_grill",]:
        pass
    elif task == "sweep_to_dustpan_of_size":
        pass
        # if "dustpan" in item['key_name'] and 'broom' not in item['key_name']:
        #     result.append(item['name2ids']['dirt0'])
    elif task == "put_money_in_safe":
        if 'safe' in item['key_name']:
            result = find_ids(['safe_body'])
            # center_body = item['pcd'][item['mask'] == item['name2ids']['safe_body']].max(axis=0)
            # center_dollar = item['pcd'][item['mask'] == item['name2ids']['dollar_stack']].max(axis=0)
            # if center_body[-1] > center_dollar[-1]:
            #     result += find_ids(['dollar_stack'])
    elif task in ["put_item_in_drawer", "open_drawer"]:
        entire_drawer = [i for i, k in item['id2names'].items() if 'drawer' in k]
        if task == 'open_drawer':
            result = entire_drawer
        else: 
            if item['key_name'] == 'drawer_frame':
                result = entire_drawer
            elif 'drawer' in item['key_name']:
                if item['kf_t'] < 200:
                    result = entire_drawer

    elif task == "reach_and_drag":
        pass
        # if item['key_name'] == 'target0':
        #     result = find_ids(['target0', 'cube'])
    elif task == "slide_block_to_color_target":
        if 'target' in item['key_name']:
            targets = ['target1', 'target2', 'target3', 'target4']
            colors = ['green', 'blue', 'pink', 'yellow']
            for i, (c, t) in enumerate(zip(colors, targets)):
                if c in desc:
                    result = [item['name2ids'][t],]
                    break
    elif task == "turn_tap":
        if 'right' in desc:
            result = find_ids(['tap_right_visual'])
        else:
            result = find_ids(['tap_left_visual'])
    elif task == "put_groceries_in_cupboard":
        for g in GROCERY_NAMES:
            if g in desc:
                g = g.replace(' ', '_') + '_visual'
                break
        if 'cupboard' not in item['key_name']:
            result = find_ids([g])
        else:
            pass
            # center_cupboard =item['pcd'][item['mask'] == item['name2ids']['cupboard']].mean(axis=0) 
            # center_obj = item['pcd'][item['mask'] == item['name2ids'][g]].mean(axis=0)
            # if abs(center_obj[-1] - center_cupboard[-1]) < 0.05:
            #     result.append(item['name2ids'][g]) 
    elif task == "place_shape_in_shape_sorter":
        for g in SHAPE_NAMES:
            if g in desc:
                g = g.replace(' ', '_')
                g = (g + '_visual') if g in ('star', 'moon') else g
                break
        if 'shape_sorter' in item['key_name']:
            result = find_ids(['shape_sorter_visual', 'shape_sorter'])
            # center_sorter = item['pcd'][item['mask'] == item['name2ids']['shape_sorter_visual']].mean(axis=0)
            # center_obj = item['pcd'][item['mask'] == item['name2ids'][g]].mean(axis=0)
            # if center_obj[-1] > center_sorter[-1]:
            #     result.append(item['name2ids'][g])
        else:
            result = find_ids([g])
    elif task == "close_jar":
        pass
        # if 'lid' not in item['key_name']:
        #     center_lid = item['pcd'][item['mask'] == item['name2ids']['jar_lid0']].mean(axis=0)[:2]
        #     center_jar = item['pcd'][item['mask'] == item['key_id']].mean(axis=0)[:2]
        #     if np.linalg.norm(center_jar - center_lid) <= 0.03:
        #         result = [item['key_id'], item['name2ids']['jar_lid0']]
    elif task == "stack_blocks":
        # ['stack_blocks_target']
        center_plane = item['pcd'][item['mask'] == item['name2ids']['stack_blocks_target_plane']].mean(axis=0)[:2]
        center_key = item['pcd'][item['mask'] == item['key_id']].mean(axis=0)[:2]
        if np.linalg.norm(center_key - center_plane) <= 0.03: # stacking mode
            names = ['stack_blocks_target_plane']
            for i in range(4):
                _n = f'stack_blocks_target{i}'
                center_block = item['pcd'][item['mask'] == item['name2ids'][_n]].mean(axis=0)[:2]
                if np.linalg.norm(center_block - center_plane) <= 0.03 and _n != item['grasp_name']:
                    names.append(_n)
            result = find_ids(names)
    elif task == "place_wine_at_rack_location":
        if "rack" in item['key_name']:
            result = find_ids(['rack_top_visual', 'rack_bottom_visual'])
            # if item['kf_t'] > 145:
            #     result.append(item['name2ids']['wine_bottle_visual'])
    elif task == "light_bulb_in":
        if 'lamp' in item['key_name']:
            center_screw = item['pcd'][item['mask'] == item['name2ids']['lamp_screw']].mean(axis=0)[:2]
            center_bulb = item['pcd'][item['mask'] == item['grasp_id']].mean(axis=0)[:2] if item['grasp_id'] != -1 else 0
            if np.linalg.norm(center_bulb - center_screw) <= 0.03:
                result = find_ids(['lamp_base', 'lamp_screw']) + [item['grasp_id'],]
            else:
                result = find_ids(['lamp_base', 'lamp_screw'])
        elif 'holder' in item['key_name']:
            bulb_name = item['key_name'].replace('_holder', '')
            center_holder = item['pcd'][item['mask'] == item['key_id']].mean(axis=0)
            center_bulb = item['pcd'][item['mask'] == item['name2ids'][bulb_name]].mean(axis=0)
            if np.linalg.norm(center_bulb - center_holder) <= 0.1:
                result = [item['key_id'], item['name2ids'][bulb_name]]
        else: # bulb0/1
            holder_name = item['key_name'].replace('bulb', 'bulb_holder')
            center_bulb = item['pcd'][item['mask'] == item['key_id']].mean(axis=0)
            center_holder = item['pcd'][item['mask'] == item['name2ids'][holder_name]].mean(axis=0)
            if np.linalg.norm(center_bulb - center_holder) <= 0.1:
                result = [item['key_id'], item['name2ids'][holder_name]]
            else:
                center_screw = item['pcd'][item['mask'] == item['name2ids']['lamp_screw']].mean(axis=0)[:2]
                if np.linalg.norm(center_bulb[:2] - center_screw) <= 0.03:
                    result = [item['key_id']] + find_ids(['lamp_base', 'lamp_screw'])
        # for k in list(result):
        #     name = item['id2names'][k] 
        #     if 'bulb' in name and 'holder' not in name:
        #         if ('light_' + name) in item['name2ids']:
        #             result.append(item['name2ids']['light_' + name])
    elif task == 'place_cups':
        if item['grasp_id'] != -1:
            if 'mug' in item['key_name'] and item['key_name'] != item['grasp_name']:
                result = [item['grasp_id']]

            if 'poke' in item['key_name'] or item['key_name'] == 'mug_visual3':
                poke_name = 'place_cups_holder_spoke' + item['grasp_name'][-1]
                cup_tree = ['place_cups_holder_base'] + [f'place_cups_holder_spoke{j}' for j in range(3)] 
                # center_poke = item['pcd'][item['mask'] == item['name2ids'][poke_name]].mean(axis=0)
                # center_grasp = item['pcd'][item['mask'] == item['grasp_id']].mean(axis=0)
                # if np.linalg.norm(center_poke - center_grasp) < 0.12:
                #     result = find_ids([item['grasp_name']] + cup_tree)
                # else:
                result = find_ids(cup_tree)
    elif task == "insert_onto_square_peg":
        pass
        # if "pillar" in item['key_name']:
        #     center_pillar = item['pcd'][item['mask'] == item['key_id']].mean(axis=0)[:2]
        #     center_ring = item['pcd'][item['mask'] == item['name2ids']['square_ring']].mean(axis=0)[:2]
        #     if np.linalg.norm(center_ring - center_pillar) <= 0.03:
        #         result = [item['key_id'], item['name2ids']['square_ring']]
    elif task == "stack_cups":
        if item['grasp_id'] != -1:
            if item['key_name'] in ['cup1_visual', 'cup2_visual']:
                center_cup1 = item['pcd'][item['mask'] == item['name2ids']['cup1_visual']].mean(axis=0)
                center_cup2 = item['pcd'][item['mask'] == item['name2ids']['cup2_visual']].mean(axis=0)
                if np.linalg.norm(center_cup1 - center_cup2) <= 0.05:
                    result = find_ids(['cup1_visual', 'cup2_visual'])
                # center_grasp = item['pcd'][item['mask'] == item['grasp_id']].mean(axis=0)
                # if center_grasp[-1] > center_cup2[-1] and np.linalg.norm(center_cup2[:2] - center_grasp[:2]) <= 0.03:
                #     if item['grasp_id'] not in result:
                #         result.append(item['grasp_id'])
    else:
        raise KeyError()
    
    return result


def get_color_position_mask(task, desc, id2names, rgb, mask, target=None, distractors=None):
    position_mask = np.full([len(mask)], fill_value=-1, dtype=np.float32)
    tags = parse_instructions(task, desc, color_only=True) # [('red', 8), ('white', 20)]
    name2ids = {v:k for k,v in id2names.items()}
    if len(tags) > 0:
        if task == 'push_buttons':
            targets = sorted([k for k in id2names.values() if 'push_buttons_target' in k])
            for i, tname in enumerate(targets):
                if i <= len(tags) - 1:
                    position_mask[mask == name2ids[tname]] = tags[i][1] 
        elif task == 'stack_blocks':
            for mask_id, name in id2names.items():
                if 'target' in name and 'target_plane' not in name:
                    position_mask[mask == mask_id] = tags[0][1]
        elif task == 'reach_and_drag':
            for mask_id, name in id2names.items():
                if 'target' in name:
                    position_mask[mask == mask_id] = tags[0][1]
        elif task == 'stack_cups':
            for mask_id, name in id2names.items():
                if 'cup2' in name:
                    position_mask[mask == mask_id] = tags[0][1]
        elif task == 'block_pyramid':
            for mask_id, name in id2names.items():
                if 'block_pyramid_block' in name:
                    position_mask[mask == mask_id] = tags[0][1]
        elif task in ['close_jar', 'insert_onto_square_peg', 'light_bulb_in']:
            position_mask[mask == name2ids[target]] = tags[0][1]
        elif task == 'slide_block_to_color_target':
            targets = ['target1', 'target2', 'target3', 'target4']
            colors = ['green', 'blue', 'pink', 'yellow']
            mask_name = dict(zip(colors, targets))[tags[0][0]]
            position_mask[mask == name2ids[mask_name]] = tags[0][1]
        else:
            raise KeyError()

    return position_mask