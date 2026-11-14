# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import numpy as np
import torch
import nvdiffrast.torch as dr
import imageio


# ----------------------------------------------------------------------------
# Vector operations
# ----------------------------------------------------------------------------

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, -1, keepdim=True)


def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2 * dot(x, n) * n - x


def length(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x, x), min=eps))  # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN


def safe_normalize(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return x / length(x, eps)


def to_hvec(x: torch.Tensor, w: float) -> torch.Tensor:
    return torch.nn.functional.pad(x, pad=(0, 1), mode='constant', value=w)


# ----------------------------------------------------------------------------
# Tonemapping
# ----------------------------------------------------------------------------

def tonemap_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f > 0.0031308,
        torch.pow(torch.clamp(f, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
        12.92 * f
    )


# ----------------------------------------------------------------------------
# sRGB color transforms
# ----------------------------------------------------------------------------

def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.0031308,
        f * 12.92,
        torch.pow(torch.clamp(f, 0.0031308), 1.0 / 2.4) * 1.055 - 0.055
    )


def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat(
        (_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]),
        dim=-1
    ) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out


def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(
        f <= 0.04045,
        f / 12.92,
        torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4)
    )


def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat(
        (_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]),
        dim=-1
    ) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out


# ----------------------------------------------------------------------------
# Displacement texture lookup
# ----------------------------------------------------------------------------

def get_miplevels(texture: np.ndarray) -> float:
    minDim = min(texture.shape[0], texture.shape[1])
    return np.floor(np.log2(minDim))


# TODO: Handle wrapping maybe
def tex_2d(tex_map: torch.Tensor, coords: torch.Tensor, filter='nearest') -> torch.Tensor:
    tex_map = tex_map[None, ...]  # Add batch dimension
    tex_map = tex_map.permute(0, 3, 1, 2)  # NHWC -> NCHW
    tex = torch.nn.functional.grid_sample(
        tex_map,
        coords[None, None, ...] * 2 - 1,
        mode=filter,
        align_corners=False
    )
    tex = tex.permute(0, 2, 3, 1)  # NCHW -> NHWC
    return tex[0, 0, ...]


# ----------------------------------------------------------------------------
# Image scaling
# ----------------------------------------------------------------------------

def scale_img_hwc(x: torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]


def scale_img_nhwc(x: torch.Tensor, size, mag='bilinear', min='area') -> torch.Tensor:
    # assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]:  # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else:  # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


def avg_pool_nhwc(x: torch.Tensor, size) -> torch.Tensor:
    y = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    y = torch.nn.functional.avg_pool2d(y, size)
    return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


# ----------------------------------------------------------------------------
# Behaves similar to tf.segment_sum
# ----------------------------------------------------------------------------

def segment_sum(data: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    # num_segments = torch.unique_consecutive(segment_ids).shape[0]
    num_segments = max(torch.unique_consecutive(segment_ids)) + 1
    # 应该使用片段id列表元素中的最大值+1作为第一个维度的长度，而不是片段id列表去重后的长度, 如果网格有孤立顶点(没有1-ring邻接点), 原来的操作会导致越界

    # Repeats ids until same dimension as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:], dtype=torch.int64, device='cuda')).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    result = torch.zeros(*shape, dtype=torch.float32, device='cuda')

    result = result.scatter_add(0, segment_ids, data)
    return result


# ----------------------------------------------------------------------------
# Projection and transformation matrix helpers.
# ----------------------------------------------------------------------------

def projection(x=0.1, n=1.0, f=50.0):  # x是摄像机视野的纵横比, n是近剪裁平面, f是远裁剪平面
    return np.array([
        [n / x, 0, 0, 0],
        [0, n / -x, 0, 0],
        [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
        [0, 0, -1, 0]
    ]).astype(np.float32)


def translate(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ]).astype(np.float32)


def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([
        [1, 0, 0, 0],
        [0, c, s, 0],
        [0, -s, c, 0],
        [0, 0, 0, 1]
    ]).astype(np.float32)


def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1]
    ]).astype(np.float32)


def scale(s):
    return np.array([
        [s, 0, 0, 0],
        [0, s, 0, 0],
        [0, 0, s, 0],
        [0, 0, 0, 1]
    ]).astype(np.float32)


def scale_xy(s):
    return np.array([
        [s, 0, 0, 0],
        [0, s, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).astype(np.float32)


def lookAt(eye, at, up):
    a = eye - at
    b = up
    w = a / np.linalg.norm(a)
    u = np.cross(b, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    translate_m = np.array([
        [1, 0, 0, -eye[0]],
        [0, 1, 0, -eye[1]],
        [0, 0, 1, -eye[2]],
        [0, 0, 0, 1]
    ]).astype(np.float32)
    rotate_m = np.array([
        [u[0], u[1], u[2], 0],
        [v[0], v[1], v[2], 0],
        [w[0], w[1], w[2], 0],
        [0, 0, 0, 1]
    ]).astype(np.float32)
    return np.matmul(rotate_m, translate_m)


def random_rotation_translation(t):
    m = np.random.normal(size=[3, 3])
    m[1] = np.cross(m[0], m[2])
    m[2] = np.cross(m[0], m[1])
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    m = np.pad(m, [[0, 1], [0, 1]], mode='constant')
    m[3, 3] = 1.0
    m[:3, 3] = np.random.uniform(-t, t, size=[3])
    return m


def center_by_screen(mvp: np.ndarray, v_pos: np.ndarray, s: float = 1.0, debug_log=False):
    """
    🌟 修复版：过滤非法 w，防止屏幕包围盒被污染
    """
    h_v_pos = np.concatenate((v_pos, np.ones((v_pos.shape[0], 1))), axis=1)  # (V, 4)
    h_v_pos_projected = (mvp @ h_v_pos.T).T  # (V, 4)
    w = h_v_pos_projected[:, 3]

    # 过滤掉相机背后或极近处的点，以及 NaN/Inf
    valid_mask = (w > 1e-4) & np.isfinite(w)

    if not np.any(valid_mask):
        if debug_log:
            print("Warning: No valid vertices found for center_by_screen, returning original mvp.")
        return mvp, 1.0

    valid_projected = h_v_pos_projected[valid_mask]
    valid_w = w[valid_mask]

    ndc = valid_projected[:, :3] / valid_w[:, np.newaxis]
    valid_ndc_mask = np.isfinite(ndc).all(axis=1)

    if not np.any(valid_ndc_mask):
        return mvp, 1.0

    ndc = ndc[valid_ndc_mask]
    xy = ndc[:, :2]

    v_min, v_max = np.amin(xy, axis=0), np.amax(xy, axis=0)
    rect_center = (v_min + v_max) * 0.5
    rect_size = np.maximum(v_max - v_min, 1e-6)

    scaling = s * min(2 / rect_size[0], 2 / rect_size[1])
    additional_m = scale_xy(scaling) @ translate(-rect_center[0], -rect_center[1], 0)

    if debug_log:
        print(f"scale: {min(2 / rect_size[0], 2 / rect_size[1])}")

    return additional_m @ mvp, scaling


# ----------------------------------------------------------------------------
# Cosine sample around a vector N
# ----------------------------------------------------------------------------

def cosine_sample(N: np.ndarray) -> np.ndarray:
    # construct local frame
    N = N / np.linalg.norm(N)

    dx0 = np.array([0, N[2], -N[1]])
    dx1 = np.array([-N[2], 0, N[0]])

    dx = dx0 if np.dot(dx0, dx0) > np.dot(dx1, dx1) else dx1
    dx = dx / np.linalg.norm(dx)
    dy = np.cross(N, dx)
    dy = dy / np.linalg.norm(dy)

    # cosine sampling in local frame
    phi = 2.0 * np.pi * np.random.uniform()
    s = np.random.uniform()
    costheta = np.sqrt(s)
    sintheta = np.sqrt(1.0 - s)

    # cartesian vector in local space
    x = np.cos(phi) * sintheta
    y = np.sin(phi) * sintheta
    z = costheta

    # local to world
    return dx * x + dy * y + N * z


# ----------------------------------------------------------------------------
# Cosine sampled light directions around the vector N
# ----------------------------------------------------------------------------

def cosine_sample_texture(res, N: np.ndarray) -> torch.Tensor:
    # construct local frame
    N = N / np.linalg.norm(N)

    dx0 = np.array([0, N[2], -N[1]])
    dx1 = np.array([-N[2], 0, N[0]])

    dx = dx0 if np.dot(dx0, dx0) > np.dot(dx1, dx1) else dx1
    dx = dx / np.linalg.norm(dx)
    dy = np.cross(N, dx)
    dy = dy / np.linalg.norm(dy)

    X = torch.tensor(dx, dtype=torch.float32, device='cuda')
    Y = torch.tensor(dy, dtype=torch.float32, device='cuda')
    Z = torch.tensor(N, dtype=torch.float32, device='cuda')

    # cosine sampling in local frame
    phi = 2.0 * np.pi * torch.rand(res, res, 1, dtype=torch.float32, device='cuda')
    s = torch.rand(res, res, 1, dtype=torch.float32, device='cuda')
    costheta = torch.sqrt(s)
    sintheta = torch.sqrt(1.0 - s)

    # cartesian vector in local space
    x = torch.cos(phi) * sintheta
    y = torch.sin(phi) * sintheta
    z = costheta

    # local to world
    return X * x + Y * y + Z * z


# ----------------------------------------------------------------------------
# Bilinear downsample by 2x.
# ----------------------------------------------------------------------------

def bilinear_downsample(x: torch.tensor) -> torch.Tensor:
    w = torch.tensor([
        [1, 3, 3, 1],
        [3, 9, 9, 3],
        [3, 9, 9, 3],
        [1, 3, 3, 1]
    ], dtype=torch.float32, device=x.device) / 64.0
    w = w.expand(x.shape[-1], 1, 4, 4)
    x = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, padding=1, stride=2, groups=x.shape[-1])
    return x.permute(0, 2, 3, 1)


# ----------------------------------------------------------------------------
# Bilinear downsample log(spp) steps
# ----------------------------------------------------------------------------

def bilinear_downsample(x: torch.tensor, spp) -> torch.Tensor:
    w = torch.tensor([
        [1, 3, 3, 1],
        [3, 9, 9, 3],
        [3, 9, 9, 3],
        [1, 3, 3, 1]
    ], dtype=torch.float32, device=x.device) / 64.0
    g = x.shape[-1]
    w = w.expand(g, 1, 4, 4)
    x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
    steps = int(np.log2(spp))
    for _ in range(steps):
        xp = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='replicate')
        x = torch.nn.functional.conv2d(xp, w, padding=0, stride=2, groups=g)
    return x.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC


# ----------------------------------------------------------------------------
# Image display function using OpenGL.
# ----------------------------------------------------------------------------

_glfw_window = None


def display_image(image, zoom=None, size=None, title=None):  # HWC
    # Import OpenGL and glfw.
    import OpenGL.GL as gl
    import glfw

    # Zoom image if requested.
    image = np.asarray(image)
    if size is not None:
        assert zoom is None
        zoom = max(1, size // image.shape[0])
    if zoom is not None:
        image = image.repeat(zoom, axis=0).repeat(zoom, axis=1)
    height, width, channels = image.shape

    # Initialize window.
    if title is None:
        title = 'Debug window'
    global _glfw_window
    if _glfw_window is None:
        glfw.init()
        _glfw_window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(_glfw_window)
        glfw.show_window(_glfw_window)
        glfw.swap_interval(0)
    else:
        glfw.make_context_current(_glfw_window)
        glfw.set_window_title(_glfw_window, title)
        glfw.set_window_size(_glfw_window, width, height)

    # Update window.
    glfw.poll_events()
    gl.glClearColor(0, 0, 0, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glWindowPos2f(0, 0)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_format = {3: gl.GL_RGB, 2: gl.GL_RG, 1: gl.GL_LUMINANCE}[channels]
    gl_dtype = {'uint8': gl.GL_UNSIGNED_BYTE, 'float32': gl.GL_FLOAT}[image.dtype.name]
    gl.glDrawPixels(width, height, gl_format, gl_dtype, image[::-1])
    glfw.swap_buffers(_glfw_window)
    if glfw.window_should_close(_glfw_window):
        return False
    return True


# ----------------------------------------------------------------------------
# Image save helper.
# ----------------------------------------------------------------------------

def save_image(fn, x: np.ndarray) -> np.ndarray:
    imageio.imwrite(fn, np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8))


def load_image(fn) -> np.ndarray:
    img = imageio.imread(fn)
    if img.dtype == np.float32:  # HDR image
        return img
    else:  # LDR image
        return img.astype(np.float32) / 255


# ----------------------------------------------------------------------------

def time_to_text(x):
    if x > 3600:
        return "%.2f h" % (x / 3600)
    elif x > 60:
        return "%.2f m" % (x / 60)
    else:
        return "%.2f s" % x


# ----------------------------------------------------------------------------

def checkerboard(width, repetitions) -> np.ndarray:
    tilesize = int(width // repetitions // 2)
    check = np.kron(
        [[1, 0] * repetitions, [0, 1] * repetitions] * repetitions,
        np.ones((tilesize, tilesize))
    ) * 0.33 + 0.33
    return np.stack((check, check, check), axis=-1)[None, ...]


# ----------------------------------------------------------------------------
# Debug view rendering: wireframe / screen bbox / 3D AABB / skeleton
# ----------------------------------------------------------------------------

def _debug_project_points(points, mvp, resolution, eps=1e-6):
    """
    Debug 专用 3D -> 2D 投影。

    注意：
    当前工程的 projection() 已经在投影矩阵中翻转 Y：
        [0, n / -x, 0, 0]

    因此这里不要再做：
        screen_y = (1 - ndc_y) * 0.5 * resolution

    否则 debug 线框会相对真实渲染上下颠倒。
    """
    points = points.detach().cpu().numpy() if torch.is_tensor(points) else points
    mvp = mvp.detach().cpu().numpy() if torch.is_tensor(mvp) else mvp

    points = np.asarray(points, dtype=np.float32)
    mvp = np.asarray(mvp, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=bool),
            np.zeros((0, 3), dtype=np.float32)
        )

    pts_h = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float32)],
        axis=1
    )

    clip = (mvp @ pts_h.T).T  # [N, 4]
    w = clip[:, 3]

    # 当前 projection/lookAt 约定下，正常在相机前方的点 w > 0。
    valid = np.isfinite(w) & (w > eps)

    ndc = np.zeros((points.shape[0], 3), dtype=np.float32)
    ndc[valid] = clip[valid, :3] / w[valid, None]

    screen_x = (ndc[:, 0] + 1.0) * 0.5 * resolution

    # 关键修正：不要再 1 - ndc_y。
    screen_y = (ndc[:, 1] + 1.0) * 0.5 * resolution

    screen = np.stack([screen_x, screen_y], axis=1).astype(np.float32)

    valid = (
        valid
        & np.all(np.isfinite(screen), axis=1)
        & np.all(np.isfinite(ndc), axis=1)
    )

    return screen, valid, ndc


def _debug_point_in_loose_screen(p, resolution, margin_scale=4.0):
    """
    避免极端离群投影坐标导致 PIL draw 非常慢。

    这里允许超出屏幕一定范围，这样仍能看到部分越界线条趋势，
    但不会因为巨大坐标把绘图拖死。
    """
    if not np.all(np.isfinite(p)):
        return False

    margin = float(resolution) * float(margin_scale)

    return (
        -margin <= p[0] <= resolution + margin
        and -margin <= p[1] <= resolution + margin
    )


def _normalize_parent_indices(parent_indices):
    """
    规范化 skeleton parent indices。

    支持：
      - None / "null" / "none" / "" -> -1
      - int / float / numeric string -> int(x)

    -1 表示 root bone，无父节点。
    """
    if parent_indices is None:
        return None

    out = []
    for x in parent_indices:
        if x is None:
            out.append(-1)
            continue

        if isinstance(x, str):
            xs = x.strip().lower()
            if xs in ("none", "null", ""):
                out.append(-1)
                continue

        out.append(int(x))

    return out


def render_debug_view(
    v_pos,
    t_pos_idx,
    mvp,
    resolution,
    joint_pos=None,
    parent_indices=None,
    draw_3d_aabb=True,
    max_wire_edges=None,
):
    """
    绘制 debug 视图。

    内容：
      - 灰白色：opt mesh wireframe
      - 亮蓝色：screen-space bbox，更接近 center_by_screen() 使用的投影 bbox
      - 深蓝色：world-space 3D AABB 投影框
      - 深绿色：skeleton parent-child bone lines
      - 亮绿色：joint points
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Warning: PIL not installed, returning blank debug view.")
        return np.zeros((resolution, resolution, 3), dtype=np.float32)

    v_pos_np = v_pos.detach().cpu().numpy() if torch.is_tensor(v_pos) else v_pos
    t_idx_np = t_pos_idx.detach().cpu().numpy() if torch.is_tensor(t_pos_idx) else t_pos_idx
    mvp_np = mvp.detach().cpu().numpy() if torch.is_tensor(mvp) else mvp

    v_pos_np = np.asarray(v_pos_np, dtype=np.float32)
    t_idx_np = np.asarray(t_idx_np, dtype=np.int64)
    mvp_np = np.asarray(mvp_np, dtype=np.float32)

    img = Image.new("RGB", (resolution, resolution), (48, 48, 48))
    draw = ImageDraw.Draw(img)

    if v_pos_np.ndim != 2 or v_pos_np.shape[1] != 3 or v_pos_np.shape[0] == 0:
        return np.array(img, dtype=np.float32) / 255.0

    if t_idx_np.ndim != 2 or t_idx_np.shape[1] != 3 or t_idx_np.shape[0] == 0:
        return np.array(img, dtype=np.float32) / 255.0

    # 只统计当前 submesh / 当前采样真正使用到的顶点。
    used_idx = np.unique(t_idx_np.reshape(-1))
    used_idx = used_idx[(used_idx >= 0) & (used_idx < v_pos_np.shape[0])]

    if used_idx.shape[0] == 0:
        return np.array(img, dtype=np.float32) / 255.0

    screen_pts, valid_pts, _ = _debug_project_points(
        v_pos_np,
        mvp_np,
        resolution
    )

    # -------------------------------------------------------------------------
    # 1. 绘制 wireframe
    # -------------------------------------------------------------------------
    edge_count = 0

    for tri in t_idx_np:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])

        if (
            i0 < 0 or i0 >= v_pos_np.shape[0]
            or i1 < 0 or i1 >= v_pos_np.shape[0]
            or i2 < 0 or i2 >= v_pos_np.shape[0]
        ):
            continue

        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            if not (valid_pts[a] and valid_pts[b]):
                continue

            p0 = screen_pts[a]
            p1 = screen_pts[b]

            if not (
                _debug_point_in_loose_screen(p0, resolution)
                and _debug_point_in_loose_screen(p1, resolution)
            ):
                continue

            draw.line(
                [(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))],
                fill=(190, 190, 190),
                width=1
            )

            edge_count += 1

            if max_wire_edges is not None and edge_count >= max_wire_edges:
                break

        if max_wire_edges is not None and edge_count >= max_wire_edges:
            break

    # -------------------------------------------------------------------------
    # 2. 绘制 screen-space bbox：用于验证 center_by_screen 类问题
    # -------------------------------------------------------------------------
    used_valid = used_idx[valid_pts[used_idx]]

    if used_valid.shape[0] > 0:
        used_screen = screen_pts[used_valid]

        xmin = float(np.min(used_screen[:, 0]))
        xmax = float(np.max(used_screen[:, 0]))
        ymin = float(np.min(used_screen[:, 1]))
        ymax = float(np.max(used_screen[:, 1]))

        # 防止极端 bbox 坐标导致 PIL 绘制过慢，但保留越界趋势。
        margin = float(resolution) * 4.0
        xmin_c = max(-margin, min(resolution + margin, xmin))
        xmax_c = max(-margin, min(resolution + margin, xmax))
        ymin_c = max(-margin, min(resolution + margin, ymin))
        ymax_c = max(-margin, min(resolution + margin, ymax))

        draw.rectangle(
            [(xmin_c, ymin_c), (xmax_c, ymax_c)],
            outline=(0, 160, 255),
            width=3
        )

    # -------------------------------------------------------------------------
    # 3. 可选：绘制 world-space 3D AABB 的投影框
    # -------------------------------------------------------------------------
    if draw_3d_aabb:
        used_v = v_pos_np[used_idx]

        vmin = used_v.min(axis=0)
        vmax = used_v.max(axis=0)

        corners = np.array([
            [vmin[0], vmin[1], vmin[2]],
            [vmax[0], vmin[1], vmin[2]],
            [vmax[0], vmax[1], vmin[2]],
            [vmin[0], vmax[1], vmin[2]],
            [vmin[0], vmin[1], vmax[2]],
            [vmax[0], vmin[1], vmax[2]],
            [vmax[0], vmax[1], vmax[2]],
            [vmin[0], vmax[1], vmax[2]],
        ], dtype=np.float32)

        c_pts, c_valid, _ = _debug_project_points(corners, mvp_np, resolution)

        box_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        for a, b in box_edges:
            if not (c_valid[a] and c_valid[b]):
                continue

            p0 = c_pts[a]
            p1 = c_pts[b]

            if not (
                _debug_point_in_loose_screen(p0, resolution)
                and _debug_point_in_loose_screen(p1, resolution)
            ):
                continue

            draw.line(
                [(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))],
                fill=(0, 0, 255),
                width=2
            )

    # -------------------------------------------------------------------------
    # 4. 绘制骨骼：joint_pos + parent_indices
    # -------------------------------------------------------------------------
    if joint_pos is not None:
        joint_np = joint_pos.detach().cpu().numpy() if torch.is_tensor(joint_pos) else joint_pos
        joint_np = np.asarray(joint_np, dtype=np.float32)

        if joint_np.ndim == 2 and joint_np.shape[1] == 3 and joint_np.shape[0] > 0:
            j_pts, j_valid, _ = _debug_project_points(
                joint_np,
                mvp_np,
                resolution
            )

            parents = _normalize_parent_indices(parent_indices)

            # 先画 parent-child 骨架线：深绿色。
            if parents is not None:
                n = min(len(parents), joint_np.shape[0])

                for child in range(n):
                    parent = parents[child]

                    # parent < 0 表示 root bone，无父节点，不画连线。
                    if parent < 0 or parent >= joint_np.shape[0]:
                        continue

                    if not (j_valid[child] and j_valid[parent]):
                        continue

                    p0 = j_pts[parent]
                    p1 = j_pts[child]

                    if not (
                        _debug_point_in_loose_screen(p0, resolution)
                        and _debug_point_in_loose_screen(p1, resolution)
                    ):
                        continue

                    draw.line(
                        [(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))],
                        fill=(0, 100, 0),
                        width=2
                    )

            # 再画关节点：亮绿色。
            r = 3

            for i in range(joint_np.shape[0]):
                if not j_valid[i]:
                    continue

                p = j_pts[i]

                if not _debug_point_in_loose_screen(p, resolution):
                    continue

                x = float(p[0])
                y = float(p[1])

                draw.ellipse(
                    [(x - r, y - r), (x + r, y + r)],
                    fill=(0, 255, 0)
                )

    return np.array(img, dtype=np.float32) / 255.0
