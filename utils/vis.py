import random
import ipyvolume as ipv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_color_map(arr, shuffle=True):
    colors = dict(mcolors.CSS4_COLORS)
    colors.pop("white")
    colors = sorted(list(colors.items()))
    colors = [c for _, c in colors]
    if shuffle:
        random.shuffle(colors)
    nums = np.unique(arr)
    return dict(zip(nums, colors))


def show_pcd(pcd, rgb=None, frame=None, mask=None, autoscale=True, autoscale_rgb="auto", frame_color=None, save=None, title="", save_kwargs={}, return_view_scale=False, vis_kwargs={}, with_axis=False):
    x, y, z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

    if isinstance(autoscale, list):
        x_min, y_min, z_min, x_max, y_max, z_max = autoscale
    else:
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        z_min, z_max = z.min(), z.max()

    if autoscale_rgb == "auto" and rgb is not None:
        autoscale_rgb = rgb.max() >= 10 or rgb.min() <= -0.5 # then it is probably 255-based color

    
    kwargs = {}
    if rgb is not None:
        kwargs["color"] = rgb
        if autoscale_rgb:
            if rgb.max() >= 10:
                kwargs["color"] = kwargs["color"].astype(float) / 255.0
            if rgb.min() <= -0.5:
                kwargs["color"] = (kwargs["color"] + 1) / 2

    if autoscale:
        # import warnings

        # with warnings.catch_warnings():
        #     warnings.filterwarnings('error')
        #     try:
        x, y, z = (x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min), (z - z_min) / (z_max - z_min)
            # except Warning as e:
            #     print(1)
    
    if mask is not None:
        fig = ipv.figure()
        cmap = get_color_map(mask)
        for mask_ind, color in cmap.items():
            inds = mask == mask_ind
            if rgb is not None:
                if isinstance(rgb, dict):
                    color = rgb[mask_ind]
                else:
                    color = rgb[inds]
            ipv.scatter(
                x[inds],
                y[inds],
                z[inds],
                size=1,
                color=color,
                description=str(mask_ind),
                **vis_kwargs
            )
    else:
        fig = ipv.quickscatter(x, y, z, size=1, description=title.split('/')[-1] or "Pointcloud", **kwargs, **vis_kwargs)

    min_v = np.array([x_min, y_min, z_min])
    max_v = np.array([x_max, y_max, z_max])
    if frame is not None:
        if isinstance(frame[0][0], np.ndarray):
            # list of frames
            if autoscale:
                frame = [[(a - min_v) / (max_v - min_v) for a in f] for f in frame]
            if isinstance(frame_color, list):
                it = zip(frame_color, frame)
            else:
                it = zip([frame_color] * len(frame), frame)
            for fc, f in it:
                draw_frame(*f, color=fc) 
        else:
            if autoscale:
                frame = [(a - min_v) / (max_v - min_v) for a in frame]
            draw_frame(*frame, color=frame_color)
            
            
    if not with_axis:
        ipv.style.use('minimal')
        
    if save:
        ipv.save(save, title=title or save, **save_kwargs)


    if return_view_scale:
        return ipv.gcc(), [min_v, max_v]
    else:
        return ipv.gcc()


# def scale_pts_for_view(pts, view_scale):
#     min_v, max_v = [a.reshape(1, -1) for a in view_scale]
#     return (pts - min_v) / (max_v - min_v)


def draw_frame(o, x, y, z, color=None):
    # draw_frame([0.1, 0.1, 0.1], [0.5, 0.1, 0.1], [0.1, 0.5, 0.1], [0.1, 0.1, 0.5])
    default_colors = [
            np.array([[255, 255, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]).astype(
            float) / 255,
            "red", "green", "blue"
        ]
    if color is None:
        colors = default_colors 
    elif isinstance(color, (float, int)):
        colors = [c * color if isinstance(c, np.ndarray) else c for c in default_colors]
    else:
        colors = [color, ] * 4

    ipv.scatter(
        np.array([o[0], x[0], y[0], z[0]]),
        np.array([o[1], x[1], y[1], z[1]]),
        np.array([o[2], x[2], y[2], z[2]]),
        color=colors[0],
        marker="sphere",
        size=3,
    )
    ipv.plot(
        np.array([o[0], x[0]]),
        np.array([o[1], x[1]]),
        np.array([o[2], x[2]]),
        color=colors[1],
    )
    ipv.plot(
        np.array([o[0], y[0]]),
        np.array([o[1], y[1]]),
        np.array([o[2], y[2]]),
        color=colors[2],
    )
    ipv.plot(
        np.array([o[0], z[0]]),
        np.array([o[1], z[1]]),
        np.array([o[2], z[2]]),
        color=colors[3],
    )


def draw_knn_point(source, endpoints, color='red', size=2):
    """source: (3), endpoints: (k,3) """

    ipv.scatter(
        np.concatenate([source[:1], endpoints[:, 0]]),
        np.concatenate([source[1:2], endpoints[:, 1]]),
        np.concatenate([source[2:], endpoints[:, 2]]),
        color=color,
        marker="sphere",
        size=3,
    )

    for ep in endpoints:
        ipv.plot(
            np.array([source[0], ep[0]]),
            np.array([source[1], ep[1]]),
            np.array([source[2], ep[2]]),
            color=color,
        )



def draw_ball(center, radius, N=100):
    x0, y0, z0 = center
    fig = ipv.gcf()
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = fig.xlim, fig.ylim, fig.zlim
    xstep, ystep, zstep = (xmax - xmin) / N, (ymax - ymin) / N, (zmax - zmin) / N
    x, y, z = np.ogrid[xmin:xmax:xstep, ymin:ymax:ystep, zmin:zmax:zstep]
    r = np.sqrt((x - x0)**2 + (y - y0)**2 + (z - z0)**2)
    r[r > radius] = 0
    ipv.volshow(r.T, extent=[(xmin, xmax), (ymin, ymax), (zmin, zmax)])



def to_named_masks(mask, id2names, key = None):
    vis_mask = np.empty((len(mask), ), dtype=object)
    for mask_id, name in id2names.items():
        tag = f'{name} ({mask_id})' 
        if mask_id == key: tag += '【*】'
        vis_mask[mask == mask_id] = tag
    return vis_mask