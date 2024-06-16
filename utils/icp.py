import open3d as o3d
from scipy.spatial.transform import Rotation
import numpy as np
from numpy.linalg import inv



def knn(query, points, k=1):
    import faiss
    if len(query) == 0 or len(points) == 0:
        raise ValueError("Found array with 0 sample(s) (shape=(0, 3)) while a minimum of 1 is required by NearestNeighbors.")
    
    index = faiss.IndexFlatL2(points.shape[1])
    index.add(points.astype(np.float32))
    distances, indices = index.search(query.astype(np.float32), k=k)
    distances = np.sqrt(distances)
    if k == 1:
        return distances.flatten(), indices.flatten()
    else:
        return distances, indices


def icp(source_pts, target_pts, source_rgb=None, target_rgb=None, max_corr_dist=0.2, max_iteration=30, rotation_hint=False, init_X=None, plane=False):
    if init_X is None:
        source_center = source_pts.mean(axis=0, keepdims=True)
        target_center = target_pts.mean(axis=0, keepdims=True)
        source = o3d.geometry.PointCloud()
        # ICP works better with normalized points
        source.points = o3d.utility.Vector3dVector(source_pts - source_center)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_pts - target_center)
        has_color = source_rgb is not None
        if has_color:
            source.colors = o3d.utility.Vector3dVector(source_rgb)
            target.colors = o3d.utility.Vector3dVector(target_rgb)

        source.estimate_normals()
        target.estimate_normals()

        H_init = np.asarray([[1, 0, 0, 0.],
                                [0, 1, 0., 0.],
                                [0, 0, 1, 0], 
                                [0.0, 0.0, 0.0, 1.0]])
        trans_init = H_init.copy()

        def register(init):
            try:
                reg_p2l = getattr(o3d.pipelines.registration, "registration_colored_icp" if has_color else "registration_icp")(
                source, target, max_corr_dist, init,
                o3d.pipelines.registration.TransformationEstimationForColoredICP()
                if has_color else
                getattr(o3d.pipelines.registration, 'TransformationEstimationPointTo' + ('Plane' if plane else 'Point'))(), 
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)) #  TransformationEstimationPointToPlane
            except RuntimeError:
                reg_p2l = None
            return reg_p2l
        
        if not rotation_hint:
            reg_p2l = register(trans_init)
        else:
            # clever initialization to improve ICP accuracy
            inits = []
            for rad in [0, np.pi/2, np.pi]:
                X = H_init.copy()
                X[:3, :3] = Rotation.from_euler('z', rad).as_matrix()
                inits.append(X)
            results = [register(init) for init in inits]
            max_fitness = max([a.fitness for a in results if a is not None] + [0])
            reg_p2l = None
            min_mse = 100
            for r in results:
                if r is not None and r.fitness == max_fitness:
                    v = r.inlier_rmse
                    if v < min_mse:
                        reg_p2l = r
                        min_mse = v
        
        if reg_p2l is not None:
            # transform X back to non-normalized space
            X = reg_p2l.transformation.copy()
            X[:3, 3] += (target_center.flatten() - (X[:3, :3] @ source_center.flatten()))
            reg_p2l.transformation = X

        return reg_p2l
    else:
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_pts)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_pts)
        has_color = source_rgb is not None
        if has_color:
            source.colors = o3d.utility.Vector3dVector(source_rgb)
            target.colors = o3d.utility.Vector3dVector(target_rgb)
        source.estimate_normals()
        target.estimate_normals()

        try:
            reg_p2l = getattr(o3d.pipelines.registration, "registration_colored_icp" if has_color else "registration_icp")(
            source, target, max_corr_dist, init_X,
            o3d.pipelines.registration.TransformationEstimationForColoredICP()
            if has_color else
            getattr(o3d.pipelines.registration, 'TransformationEstimationPointTo' + ('Plane' if plane else 'Point'))(), 
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)) #  TransformationEstimationPointToPlane
        except RuntimeError:
            reg_p2l = None
        return reg_p2l


def estimate_pca_box(pcd):
    return to_o3d_pcd(pcd).get_minimal_oriented_bounding_box()


def to_o3d_pcd(pcd):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(pcd)
    return source


def to_np_pcd(o3d_pcd):
    return np.asarray(o3d_pcd.points)


def box_volume(pcd):
    try:
        return estimate_pca_box(pcd).volume()
    except Exception as e:
        msg = str(e)
        print(f'Exception while estimating point cloud volume: {msg}, return zero volume')
        return 0.0


def voxel_grid(pcd, voxel_size=0.02):
    source = to_o3d_pcd(pcd)
    voxel_volume = o3d.geometry.VoxelGrid.create_from_point_cloud(source, voxel_size)
    return np.asarray([voxel_volume.origin + pt.grid_index*voxel_volume.voxel_size for pt in voxel_volume.get_voxels()])


def normalize_point_cloud_to_origin(pcd):
    center = pcd.mean(axis=0, keepdims=True)
    pcd = pcd - center
    return pcd, center.flatten()


def to_unit_length(v):
    return v / np.linalg.norm(v)


def R_2_X(R):
    X = np.eye(4)
    X[:3, :3] = R
    return X

    
def t_2_X(t):
    X = np.eye(4)
    if isinstance(t, np.ndarray): t = t.flatten()
    X[:3, -1] = t
    return X


def Rt_2_X(R, t):
    X = np.eye(4)
    X[:3, :3] = R
    if isinstance(t, np.ndarray): t = t.flatten()
    X[:3, -1] = t
    return X

def X_2_Rt(X):
    return X[:3, :3], X[:3, -1]

def to_homo_axis(pts):
    return np.concatenate([pts, np.ones((len(pts), 1))], axis=1)


def h_transform(T, pts):
    return (T @ to_homo_axis(pts).T).T[:, :3]

def r_transform(R, pts):
    return (R @ pts.T).T

def axis_angle_rotate(axis, radian):
    assert axis.shape == (3,)
    rot = Rotation.from_rotvec(to_unit_length(axis) * radian)
    return rot.as_matrix() 

def rotate_X(X, Pc, axis, angle):
    R, t = X_2_Rt(X)
    Rot_inv = inv(axis_angle_rotate(axis, angle))
    Rnew = Rot_inv @ R
    tnew = Rot_inv @ (t - Pc) + Pc
    return Rt_2_X(Rnew, tnew)


def get_matching_ratio(P_from, P_to, threshold=0.01):
    dist, _ = knn(P_from, P_to)
    matched_ratio = (dist <= threshold).sum() / len(P_from)
    return matched_ratio


def resolve_rotation_ambiguity(X, ref_points, points, ref_context_points, context_points, ambiguity_threshold=0.95):
    P = points
    P_n, P_c = normalize_point_cloud_to_origin(P)
    bbox = estimate_pca_box(P_n)
    
    P_ref = h_transform(X, ref_points)
    P_ref_n, P_ref_c = normalize_point_cloud_to_origin(P_ref)
    
    X_alternatives = [X, ]
    X_AXIS, Y_AXIS, Z_AXIS = 0, 1, 2

    for radian in [np.pi, np.pi/2, -np.pi/2]:
        for axis in [X_AXIS, Y_AXIS, Z_AXIS]:
            ratio = get_matching_ratio(P_ref_n, r_transform(axis_angle_rotate(bbox.R[axis], radian), P_n))
            # print('ambiguity ratio -> ', ratio)
            if ratio >= ambiguity_threshold:
                _X = rotate_X(X, P_c, bbox.R[axis], radian)
                if check_X_validity(_X):
                    X_alternatives.append(_X)
    
    if len(X_alternatives) == 1:
        return X_alternatives[0]
    
    distances = []
    Xs = []
    for _X in X_alternatives:
        distance = knn(h_transform(_X, ref_context_points), context_points)[0].mean()
        distances.append(distance)
        Xs.append(_X)
    
    ind = np.argmin(distances)
    # if distances[0] / (distances[ind] + 1e-6) > 2: 
    if ind != 0:
        print('ambiguity resolve to another X!')
    else:
        ind = 0
    chosen_X = Xs[ind]
    return chosen_X
    
    
    
def check_X_validity(X):
    plane = np.zeros((10, 10, 3))
    plane[:, :, 0], plane[:, :, 1] = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    plane[:, :, 2] = np.random.uniform(0.74, 0.76, size=(10, 10))
    plane = plane.reshape(-1, 3)
    X_plane = h_transform(X, plane)
    m = X_plane[:, 2].mean()
    return 0.73 <=  m <= 0.77


def pose7_to_X(pose):
    R = Rotation.from_quat(pose[3:]).as_matrix()
    t = pose[:3]
    X = np.zeros((4, 4))
    X[:3, :3] = R
    X[-1, -1] = 1
    X[:3, -1] = t
    return X


def X_to_pose7(X):
    t = X[:3, -1]
    q = Rotation.from_matrix(X[:3, :3]).as_quat()
    return np.concatenate([t, q])


def pose7_to_frame(pose, scale=0.1):
    pose = pose.copy()
    R = Rotation.from_quat(pose[3:]).as_matrix() * scale
    t = pose[:3]
    return np.array([t, R[0] + t, R[1] + t, R[2] + t])


def X_to_frame(X):
    return pose7_to_frame(X_to_pose7(X))


def frame_to_X(frame):
    frame = np.copy(frame)
    t, x, y, z = frame
    X = np.eye(4)
    X[:3, -1] = t
    x -= t
    y -= t
    z -= t
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)
    X[0, :3] = x
    X[1, :3] = y
    X[2, :3] = z
    return X


def h_transform_X(T, X):
    return frame_to_X(h_transform(T, np.array(X_to_frame(X))))

def h_transform_pose(T, pose):
    return X_to_pose7(frame_to_X(h_transform(T, np.array(pose7_to_frame(pose)))))
    

def rotate_from_origin(pts, matrix):
    center = pts.mean(axis=0, keepdims=True) 
    pts -= center
    pts = r_transform(matrix, pts)
    pts += center
    return pts
    

def fps_sample_to(pts, N):
    if N >= len(pts):
        return pts
    else:
        return to_np_pcd(to_o3d_pcd(pts).farthest_point_down_sample(N))
    
    
    
def arun(A, B):
    """Solve 3D registration using Arun's method: B = RA + t
    """
    A, B = A.T, B.T
    
    N = A.shape[1]
    assert B.shape[1] == N

    # calculate centroids
    A_centroid = np.reshape(1/N * (np.sum(A, axis=1)), (3,1))
    B_centroid = np.reshape(1/N * (np.sum(B, axis=1)), (3,1))

    # calculate the vectors from centroids
    A_prime = A - A_centroid
    B_prime = B - B_centroid

    # rotation estimation
    H = np.zeros([3, 3])
    for i in range(N):
        ai = A_prime[:, i]
        bi = B_prime[:, i]
        H = H + np.outer(ai, bi)
    U, S, V_transpose = np.linalg.svd(H)
    V = np.transpose(V_transpose)
    U_transpose = np.transpose(U)
    R = V @ np.diag([1, 1, np.linalg.det(V) * np.linalg.det(U_transpose)]) @ U_transpose

    # translation estimation
    t = B_centroid - R @ A_centroid

    return R, t.flatten()