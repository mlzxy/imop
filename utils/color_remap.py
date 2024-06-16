import re
import numpy as np
from scipy import stats
import itertools

def filter_products(pds):
    out = set()
    for a in pds:
        if len(a) == len(set(a)):
            out.add(tuple(sorted(a)))
    return list(out)


def generate_object_pairs(mask_ids, k):
    return filter_products(itertools.product([int(k) for k in mask_ids], repeat=k))


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
    ('lightgray', (0.35, 0.35, 0.35)),
    ('orange', (1.0, 0.5, 0.0)),
    ('olive', (0.5, 0.5, 0.0)),
    ('purple', (0.5, 0.0, 0.5)),
    # ('pink', (0.95, 0.075, 0.54)),
    ('teal', (0, 0.5, 0.5)),
    ('azure', (0.0, 0.5, 1.0)),
    ('violet', (0.5, 0.0, 1.0)),
    ('rose', (1.0, 0.0, 0.5)),
    ('black', (0.0, 0.0, 0.0)),
    ('white', (1.0, 1.0, 1.0)),
])

# colors = {'blue': [0.211, 0.348, 0.699],
#  'teal': [0.211, 0.606, 0.459],
#  'lime': [0.21, 0.842, 0.203],
#  'yellow': [0.704, 0.842, 0.202],
#  'navy': [0.204, 0.34, 0.458],
#  'orange': [0.701, 0.606, 0.204],
#  'purple': [0.465, 0.362, 0.457],
#  'white': [0.704, 0.844, 0.699],
#  'green': [0.21, 0.608, 0.203],
#  'azure': [0.21, 0.602, 0.693],
#  'olive': [0.465, 0.602, 0.2],
#  'rose': [0.702, 0.362, 0.461],
#  'red': [0.709, 0.341, 0.199],
#  'cyan': [0.214, 0.843, 0.694],
#  'gray': [0.467, 0.606, 0.46],
#  'silver': [0.601, 0.732, 0.595],
#  'maroon': [0.471, 0.354, 0.206],
#  'black': [0.208, 0.346, 0.201],
#  'magenta': [0.701, 0.356, 0.692],
#  'violet': [0.469, 0.347, 0.7]}


# for k in list(colors.keys()):
#     v = colors[k]
#     colors[k] = (v[0] * 0.8, v[1] * 0.8, v[2] * 0.8)

color_names = []
color_values = []

for c, cv in colors.items():
    color_names.append(c) 
    color_values.append(cv)

color_names = np.array(color_names)
color_values = np.array(color_values)


def find_color_directive(desc):
    cs = []
    for c in colors:
        c = f' {c} '
        if c in desc:
            cs.append((desc.index(c), c.strip()))
    
    cs = sorted(cs)
    return [c for _, c in cs]


def remap_colors(task, rgb, new_mask, id2names, desc):
    other_color = 'blue'
    name2ids = {v: k for k,v in id2names.items()}

    if task == 'slide_block_to_color_target':
        return rgb
    elif task == 'push_buttons':
        changes = [(e, ['red', 'green', 'blue'][i]) for i, e in enumerate(find_color_directive(desc))]
        targets = sorted([k for k in id2names.values() if 'push_buttons_target' in k])
        for t, (_, cto) in zip(targets, changes):
            rgb[new_mask == name2ids[t]] = (np.array(colors[cto]) * 255).astype(int)           
        return rgb
    elif task == 'stack_blocks':
        for mask_id, name in id2names.items():
            if 'target' in name and 'target_plane' not in name:
                rgb[new_mask == mask_id] = np.array([255, 0, 0])
            elif 'distractor' in name:
                rgb[new_mask == mask_id] = (np.array(colors[other_color]) * 255).astype(int)
        return rgb
    elif task == 'reach_and_drag':
        for mask_id, name in id2names.items():
            if 'target' in name:
                rgb[new_mask == mask_id] = np.array([255, 0, 0])
            elif 'distractor' in name:
                rgb[new_mask == mask_id] = (np.array(colors[other_color]) * 255).astype(int)
        return rgb
    else:
        targets = []
        changes = {}
        for k in id2names.values():
            if 'target' in k or \
            'distractor' in k or \
            re.match(r'jar\d', k) or \
            re.match(r'cup\d_visual', k) or \
            'bulb_holder' in k or \
            'pillar' in k:
                existing_colors = find_color_directive(desc)
                assert len(existing_colors) == 1, desc
                changes[existing_colors[0]] = 'red'
                targets.append(k)
    
    if len(targets) > 0:
        def norm(c):
            return c / np.linalg.norm(c)

        def find_by_color(cfrom):
            # could not handle black color!
            if cfrom != 'black':
                color_sims = [(np.dot( norm((rgb[new_mask == name2ids[t]] / 255.).mean(axis=0)), norm(np.array(colors[cfrom]))), t)  for t in targets]
                color_sims = sorted(color_sims, reverse=True)
                remaining_colors = [color_sims[0][1]] + [c for v, c in color_sims[1:] if abs((color_sims[0][0] - v) / color_sims[0][0]) < 0.03]
                color_diffs = [(np.sum(np.abs((rgb[new_mask == name2ids[t]] / 255.).mean(axis=0) - np.array(colors[cfrom]))), t)  for t in remaining_colors]           
                t = min(color_diffs)[1]       
            else:
                color_diffs = [(np.sum(np.abs((rgb[new_mask == name2ids[t]] / 255.).mean(axis=0) - np.array(colors[cfrom]))), t)  for t in targets]           
                t = min(color_diffs)[1]                      
            return t
        # singular changes
        for cfrom, cto in changes.items():
            t = find_by_color(cfrom)
            if cfrom == 'gray':
                t2 = find_by_color('lightgray')
                if t2 != t: t = t2
            rgb[new_mask == name2ids[t]] = (np.array(colors[cto]) * 255).astype(int)           
            targets.remove(t)

        for t in targets:
            rgb[new_mask == name2ids[t]] = (np.array(colors[other_color]) * 255).astype(int)           

    return rgb
            

