# filepath: src/ogd_lbs/diff_render/camera.py
import torch
import numpy as np
from .config import DiffRenderConfig
from ogd_lbs.diff_render.nvdiffmodeling import util, mesh


# ======================================================================================
# Legacy NumPy helpers
# ======================================================================================
def get_used_vertices(m: mesh.Mesh):
    """提取网格中实际被面片引用的顶点。Legacy preview/debug 路径使用。"""
    v_pos_np = m.v_pos.detach().cpu().numpy()
    t_pos_idx_np = m.t_pos_idx.detach().cpu().numpy()
    if t_pos_idx_np.size == 0:
        return v_pos_np
    return v_pos_np[np.unique(t_pos_idx_np.flatten())]


# ======================================================================================
# Torch camera helpers for training path
# ======================================================================================
def _to_batch_scalar(x, batch_size: int, device, dtype=torch.float32):
    if torch.is_tensor(x):
        x = x.to(device=device, dtype=dtype)
        if x.ndim == 0:
            return x.expand(batch_size)
        return x.reshape(batch_size)

    return torch.full(
        (batch_size,),
        float(x),
        dtype=dtype,
        device=device
    )


def projection_torch(
    batch_size: int,
    device,
    dtype=torch.float32,
    x: float = 0.005,
    n: float = 0.01,
    f: float = 1000.0,
):
    """
    Torch version of util.projection().

    对齐原始 util.projection(x=0.005, n=0.01, f=1000.0)：

        [[n / x, 0, 0, 0],
         [0, n / -x, 0, 0],
         [0, 0, -(f+n)/(f-n), -(2fn)/(f-n)],
         [0, 0, -1, 0]]
    """
    p = torch.zeros(
        (batch_size, 4, 4),
        dtype=dtype,
        device=device
    )

    p[:, 0, 0] = n / x
    p[:, 1, 1] = n / -x
    p[:, 2, 2] = -(f + n) / (f - n)
    p[:, 2, 3] = -(2.0 * f * n) / (f - n)
    p[:, 3, 2] = -1.0

    return p


def translate_torch(x, y, z, batch_size: int, device, dtype=torch.float32):
    """
    Batched Torch version of util.translate().
    """
    m = torch.eye(
        4,
        dtype=dtype,
        device=device
    ).expand(batch_size, 4, 4).clone()

    x = _to_batch_scalar(x, batch_size, device, dtype)
    y = _to_batch_scalar(y, batch_size, device, dtype)
    z = _to_batch_scalar(z, batch_size, device, dtype)

    m[:, 0, 3] = x
    m[:, 1, 3] = y
    m[:, 2, 3] = z

    return m


def scale_xy_torch(s, batch_size: int, device, dtype=torch.float32):
    """
    Batched Torch version of util.scale_xy().
    """
    m = torch.eye(
        4,
        dtype=dtype,
        device=device
    ).expand(batch_size, 4, 4).clone()

    s = _to_batch_scalar(s, batch_size, device, dtype)

    m[:, 0, 0] = s
    m[:, 1, 1] = s

    return m


def look_at_torch(eye: torch.Tensor, at: torch.Tensor, up: torch.Tensor):
    """
    Torch version of util.lookAt().

    eye: [B, 3]
    at:  [B, 3]
    up:  [B, 3]

    对齐原始 util.lookAt()：

        a = eye - at
        b = up
        w = normalize(a)
        u = normalize(cross(b, w))
        v = cross(w, u)

        lookAt = rotate @ translate

    返回：
        [B, 4, 4]
    """
    w = torch.nn.functional.normalize(
        eye - at,
        dim=-1,
        eps=1e-8
    )

    u = torch.nn.functional.normalize(
        torch.cross(up, w, dim=-1),
        dim=-1,
        eps=1e-8
    )

    v = torch.cross(w, u, dim=-1)

    batch_size = eye.shape[0]
    m = torch.eye(
        4,
        dtype=eye.dtype,
        device=eye.device
    ).expand(batch_size, 4, 4).clone()

    m[:, 0, :3] = u
    m[:, 1, :3] = v
    m[:, 2, :3] = w

    # rotate @ translate
    m[:, 0, 3] = -torch.sum(u * eye, dim=-1)
    m[:, 1, 3] = -torch.sum(v * eye, dim=-1)
    m[:, 2, 3] = -torch.sum(w * eye, dim=-1)

    return m


def get_used_vertices_torch(m: mesh.Mesh):
    """
    Torch version of get_used_vertices().

    训练路径中必须避免：
        m.v_pos.detach().cpu().numpy()
        m.t_pos_idx.detach().cpu().numpy()

    注意：
      - 返回 detach 后的顶点，保持旧 camera 逻辑中 camera 不对 mesh 反传的语义。
      - sub_mesh 当前只裁剪 t_pos_idx，不裁剪 v_pos，因此这里使用 t_pos_idx 中真实引用的顶点。
    """
    v_pos = m.v_pos.detach()
    t_pos_idx = m.t_pos_idx

    if t_pos_idx is None or t_pos_idx.numel() == 0:
        return v_pos

    used_idx = torch.unique(t_pos_idx.reshape(-1))
    used_idx = used_idx[
        (used_idx >= 0)
        & (used_idx < v_pos.shape[0])
    ]

    if used_idx.numel() == 0:
        return v_pos

    return v_pos[used_idx]


def center_by_screen_torch(
    mvp: torch.Tensor,
    v_pos: torch.Tensor,
    s=0.9,
    eps: float = 1e-4,
):
    """
    Torch version of util.center_by_screen().
    🌟 修复版：过滤非法 w，防止屏幕包围盒被污染
    """
    batch_size = mvp.shape[0]
    device = mvp.device
    dtype = mvp.dtype

    if v_pos is None or v_pos.numel() == 0:
        scaling = torch.ones(
            batch_size,
            dtype=dtype,
            device=device
        )
        return mvp, scaling

    v_pos = v_pos.to(device=device, dtype=dtype)

    ones = torch.ones(
        (v_pos.shape[0], 1),
        dtype=dtype,
        device=device
    )

    h_v_pos = torch.cat(
        [v_pos, ones],
        dim=-1
    )  # [N, 4]

    clip = torch.matmul(
        h_v_pos[None, :, :],
        mvp.transpose(1, 2)
    )  # [B, N, 4]

    w = clip[..., 3]

    # 过滤掉相机背后或极近处的点，以及 NaN/Inf
    valid = (w > eps) & torch.isfinite(w)
    valid_any = torch.any(valid, dim=1)

    w_safe = torch.where(
        valid,
        w,
        torch.full_like(w, eps)
    )

    ndc = clip[..., :3] / w_safe[..., None]
    xy = ndc[..., :2]

    # 仅计算 valid 点的 min/max
    inf = torch.full_like(xy, float("inf"))
    neg_inf = torch.full_like(xy, float("-inf"))

    xy_min = torch.where(
        valid[..., None],
        xy,
        inf
    ).amin(dim=1)

    xy_max = torch.where(
        valid[..., None],
        xy,
        neg_inf
    ).amax(dim=1)

    # 对没有 valid 的 batch 给默认值，避免中间 inf 影响后续矩阵
    xy_min = torch.where(
        valid_any[:, None],
        xy_min,
        torch.zeros_like(xy_min)
    )

    xy_max = torch.where(
        valid_any[:, None],
        xy_max,
        torch.ones_like(xy_max)
    )

    rect_center = (xy_min + xy_max) * 0.5
    rect_size = torch.clamp(xy_max - xy_min, min=1e-6)

    s_batch = _to_batch_scalar(s, batch_size, device, dtype)

    scaling = s_batch * torch.minimum(
        2.0 / rect_size[:, 0],
        2.0 / rect_size[:, 1]
    )

    scale_m = scale_xy_torch(
        scaling,
        batch_size=batch_size,
        device=device,
        dtype=dtype
    )

    trans_m = translate_torch(
        -rect_center[:, 0],
        -rect_center[:, 1],
        torch.zeros(batch_size, dtype=dtype, device=device),
        batch_size=batch_size,
        device=device,
        dtype=dtype
    )

    additional_m = torch.matmul(scale_m, trans_m)
    revised_mvp = torch.matmul(additional_m, mvp)

    return revised_mvp, scaling


def _focus_auto_center_if_inside_torch(
    config: DiffRenderConfig,
    base_mvp: torch.Tensor,
    all_v_pos: torch.Tensor,
):
    """
    Torch version of focus-mode intelligent auto-center logic.
    """
    if not getattr(config, 'enable_auto_center', True):
        return base_mvp

    if all_v_pos is None or all_v_pos.numel() == 0:
        return base_mvp

    batch_size = base_mvp.shape[0]
    device = base_mvp.device
    dtype = base_mvp.dtype

    all_v_pos = all_v_pos.to(device=device, dtype=dtype)

    ones = torch.ones(
        (all_v_pos.shape[0], 1),
        dtype=dtype,
        device=device
    )
    h = torch.cat([all_v_pos, ones], dim=-1)  # [N, 4]

    clip = torch.matmul(
        h[None, :, :],
        base_mvp.transpose(1, 2)
    )  # [B, N, 4]

    w = clip[..., 3]
    valid = (w > 1e-4) & torch.isfinite(w)
    valid_any = torch.any(valid, dim=1)

    w_safe = torch.where(
        valid[..., None],
        w[..., None],
        torch.full_like(w[..., None], 1e-4)
    )

    ndc = clip[..., :3] / w_safe
    xy = ndc[..., :2]

    inside_xy = (
        (xy[..., 0] >= -1.0)
        & (xy[..., 0] <= 1.0)
        & (xy[..., 1] >= -1.0)
        & (xy[..., 1] <= 1.0)
    )

    # invalid 点忽略，相当于旧逻辑中只对 valid_mask 子集检查。
    all_inside_valid = torch.all(
        torch.where(valid, inside_xy, torch.ones_like(inside_xy)),
        dim=1
    )

    should_center = valid_any & all_inside_valid

    # 对 valid 点计算 bbox。
    inf = torch.full_like(xy, float("inf"))
    neg_inf = torch.full_like(xy, float("-inf"))

    xy_min = torch.where(
        valid[..., None],
        xy,
        inf
    ).amin(dim=1)

    xy_max = torch.where(
        valid[..., None],
        xy,
        neg_inf
    ).amax(dim=1)

    # 对没有 valid 的 batch 给默认值，避免中间 inf 影响后续矩阵。
    xy_min = torch.where(
        valid_any[:, None],
        xy_min,
        torch.zeros_like(xy_min)
    )

    xy_max = torch.where(
        valid_any[:, None],
        xy_max,
        torch.ones_like(xy_max)
    )

    center = (xy_min + xy_max) * 0.5
    size = torch.clamp(
        torch.max(xy_max - xy_min, dim=1).values * 0.5,
        min=1e-5
    )

    scale = 0.9 / size

    scale_m = scale_xy_torch(
        scale,
        batch_size=batch_size,
        device=device,
        dtype=dtype
    )

    trans_m = translate_torch(
        -center[:, 0],
        -center[:, 1],
        torch.zeros(batch_size, dtype=dtype, device=device),
        batch_size=batch_size,
        device=device,
        dtype=dtype
    )

    additional_m = torch.matmul(scale_m, trans_m)
    centered_mvp = torch.matmul(additional_m, base_mvp)

    revised_mvp = torch.where(
        should_center[:, None, None],
        centered_mvp,
        base_mvp
    )

    return revised_mvp


def cosine_sample_torch(N: torch.Tensor):
    """
    Batched Torch version of util.cosine_sample().

    N: [B, 3]

    返回：
        [B, 3]
    """
    N = torch.nn.functional.normalize(
        N,
        dim=-1,
        eps=1e-8
    )

    zeros = torch.zeros_like(N[:, 0])

    dx0 = torch.stack(
        [
            zeros,
            N[:, 2],
            -N[:, 1],
        ],
        dim=-1
    )

    dx1 = torch.stack(
        [
            -N[:, 2],
            zeros,
            N[:, 0],
        ],
        dim=-1
    )

    dx0_len2 = torch.sum(dx0 * dx0, dim=-1)
    dx1_len2 = torch.sum(dx1 * dx1, dim=-1)

    use_dx0 = dx0_len2 > dx1_len2
    dx = torch.where(
        use_dx0[:, None],
        dx0,
        dx1
    )

    dx = torch.nn.functional.normalize(
        dx,
        dim=-1,
        eps=1e-8
    )

    dy = torch.cross(N, dx, dim=-1)
    dy = torch.nn.functional.normalize(
        dy,
        dim=-1,
        eps=1e-8
    )

    batch_size = N.shape[0]
    device = N.device
    dtype = N.dtype

    phi = 2.0 * np.pi * torch.rand(
        batch_size,
        1,
        dtype=dtype,
        device=device
    )

    s = torch.rand(
        batch_size,
        1,
        dtype=dtype,
        device=device
    )

    costheta = torch.sqrt(s)
    sintheta = torch.sqrt(torch.clamp(1.0 - s, min=0.0))

    x = torch.cos(phi) * sintheta
    y = torch.sin(phi) * sintheta
    z = costheta

    return dx * x + dy * y + N * z


@torch.no_grad()
def get_camera_params_torch(
    config: DiffRenderConfig,
    opt_mesh: mesh.Mesh,
    ref_mesh: mesh.Mesh,
    mode: str = "train_orbit",
):
    """
    Torch/GPU camera path for training.

    支持：
      - train_orbit
      - train_focus

    设计目标：
      - 不执行 opt_mesh/ref_mesh 的 .cpu().numpy()
      - 保持 camera 对 mesh 顶点 detach，不对相机参数反传
      - 保持原始 camera 逻辑和矩阵约定
    """
    assert mode.startswith("train"), f"get_camera_params_torch only supports train modes, got: {mode}"

    device = opt_mesh.v_pos.device
    dtype = torch.float32
    batch_size = config.batch

    use_focus_mode = "focus" in mode

    proj_mtx = projection_torch(
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        x=0.005,
        n=0.01,
        f=1000.0
    )

    radius = float(config.camera_radius)

    bmesh_v_pos_used = get_used_vertices_torch(opt_mesh)
    rmesh_v_pos_used = get_used_vertices_torch(ref_mesh)

    if bmesh_v_pos_used.numel() == 0:
        all_v_pos = rmesh_v_pos_used
    elif rmesh_v_pos_used.numel() == 0:
        all_v_pos = bmesh_v_pos_used
    else:
        all_v_pos = torch.cat(
            [bmesh_v_pos_used, rmesh_v_pos_used],
            dim=0
        )

    # -------------------------------------------------------------------------
    # Commit 5: Torch focus camera.
    # -------------------------------------------------------------------------
    if use_focus_mode:
        v_pos = opt_mesh.v_pos.detach()
        t_pos_idx = opt_mesh.t_pos_idx

        if t_pos_idx is None or t_pos_idx.shape[0] == 0:
            targets = torch.zeros(
                (batch_size, 3),
                dtype=dtype,
                device=device
            )

            normals = torch.tensor(
                [0.0, 0.0, 1.0],
                dtype=dtype,
                device=device
            )[None, :].expand(batch_size, 3)
        else:
            v0 = v_pos[t_pos_idx[:, 0]]
            v1 = v_pos[t_pos_idx[:, 1]]
            v2 = v_pos[t_pos_idx[:, 2]]

            e1 = v1 - v0
            e2 = v2 - v0
            cross = torch.cross(e1, e2, dim=-1)
            area = 0.5 * torch.norm(cross, dim=-1)

            area_sum = area.sum()
            if area_sum <= 1e-6:
                probs = torch.ones_like(area) / max(1, area.shape[0])
            else:
                probs = area / area_sum

            sampled_idx = torch.multinomial(
                probs,
                batch_size,
                replacement=True
            )

            v0_s = v0[sampled_idx]
            v1_s = v1[sampled_idx]
            v2_s = v2[sampled_idx]
            cross_s = cross[sampled_idx]

            normals = torch.nn.functional.normalize(
                cross_s,
                dim=-1,
                eps=1e-8
            )

            r1 = torch.rand(
                batch_size,
                1,
                dtype=dtype,
                device=device
            )
            r2 = torch.rand(
                batch_size,
                1,
                dtype=dtype,
                device=device
            )

            sqrt_r1 = torch.sqrt(r1)
            u = 1.0 - sqrt_r1
            v = sqrt_r1 * (1.0 - r2)
            w = sqrt_r1 * r2

            targets = u * v0_s + v * v1_s + w * v2_s

        fluct_in = float(config.cam_focus_fluctuation_in)
        fluct_out = float(config.cam_focus_fluctuation_out)

        rand_dist = torch.rand(
            batch_size,
            1,
            dtype=dtype,
            device=device
        ) * (fluct_in + fluct_out) + (1.0 - fluct_in)

        D = torch.clamp(
            float(config.cam_focus_offset) * rand_dist,
            min=0.05
        )

        eyes = targets + normals * D

        up_base = torch.tensor(
            config.camera_up,
            dtype=dtype,
            device=device
        )[None, :].expand(batch_size, 3)

        dot_nu = torch.abs(
            torch.sum(normals * up_base, dim=-1, keepdim=True)
        )

        alt_up = torch.nn.functional.normalize(
            up_base + torch.tensor(
                [0.001, 0.0, 0.0],
                dtype=dtype,
                device=device
            )[None, :],
            dim=-1,
            eps=1e-8
        )

        up_final = torch.where(
            dot_nu > 0.999,
            alt_up,
            up_base
        )

        a_mv = look_at_torch(
            eyes,
            targets,
            up_final
        )

        base_mvp = torch.matmul(
            proj_mtx,
            a_mv
        )

        revised_mvp = _focus_auto_center_if_inside_torch(
            config=config,
            base_mvp=base_mvp,
            all_v_pos=all_v_pos
        )

        if config.cam_jitter_pixels > 0.0:
            pixel_size = 2.0 / float(config.train_res)
            jitter_range = float(config.cam_jitter_pixels) * pixel_size

            jitter_x = (
                torch.rand(batch_size, dtype=dtype, device=device) * 2.0 - 1.0
            ) * jitter_range

            jitter_y = (
                torch.rand(batch_size, dtype=dtype, device=device) * 2.0 - 1.0
            ) * jitter_range

            jitter_matrix = translate_torch(
                jitter_x,
                jitter_y,
                torch.zeros(batch_size, dtype=dtype, device=device),
                batch_size=batch_size,
                device=device,
                dtype=dtype
            )

            revised_mvp = torch.matmul(
                jitter_matrix,
                revised_mvp
            )

        campos = eyes
        lightpos = cosine_sample_torch(eyes) * radius

        return revised_mvp, campos, lightpos

    # -------------------------------------------------------------------------
    # Commit 4: Torch orbit camera.
    # -------------------------------------------------------------------------
    base_mvp_list = []
    campos_np_list = []
    lightpos_np_list = []

    for _ in range(batch_size):
        r_rot = util.random_rotation_translation(0.25)
        a_mv_np = np.matmul(
            util.translate(0, 0, -radius),
            r_rot
        ).astype(np.float32)

        c_pos_np = np.linalg.inv(a_mv_np)[:3, 3].astype(np.float32)

        base_mvp_np = np.matmul(
            util.projection(x=0.005, n=0.01, f=1000.0),
            a_mv_np
        ).astype(np.float32)

        base_mvp_list.append(base_mvp_np)
        campos_np_list.append(c_pos_np)
        lightpos_np_list.append(util.cosine_sample(c_pos_np) * radius)

    base_mvp = torch.tensor(
        np.stack(base_mvp_list, axis=0),
        dtype=dtype,
        device=device
    )

    campos = torch.tensor(
        np.stack(campos_np_list, axis=0),
        dtype=dtype,
        device=device
    )

    lightpos = torch.tensor(
        np.stack(lightpos_np_list, axis=0),
        dtype=dtype,
        device=device
    )

    if float(config.cam_zoom_max) > 0.0:
        zoom_np = np.random.uniform(
            1.0,
            1.0 + float(config.cam_zoom_max),
            size=(batch_size,)
        ).astype(np.float32)

        zoom = torch.tensor(
            zoom_np,
            dtype=dtype,
            device=device
        )

        s_factor = 0.9 * zoom

        if getattr(config, 'enable_auto_center', True):
            revised_mvp, _ = center_by_screen_torch(
                base_mvp,
                all_v_pos,
                s=s_factor
            )
        else:
            scaling_matrix = scale_xy_torch(
                s_factor,
                batch_size=batch_size,
                device=device,
                dtype=dtype
            )
            revised_mvp = torch.matmul(
                scaling_matrix,
                base_mvp
            )

        pan_range = (zoom - 1.0) * 0.5

        pan_x = (
            torch.rand(batch_size, dtype=dtype, device=device) * 2.0 - 1.0
        ) * pan_range

        pan_y = (
            torch.rand(batch_size, dtype=dtype, device=device) * 2.0 - 1.0
        ) * pan_range

        pan_matrix = translate_torch(
            pan_x,
            pan_y,
            torch.zeros(batch_size, dtype=dtype, device=device),
            batch_size=batch_size,
            device=device,
            dtype=dtype
        )

        revised_mvp = torch.matmul(
            pan_matrix,
            revised_mvp
        )
    else:
        if getattr(config, 'enable_auto_center', True):
            revised_mvp, _ = center_by_screen_torch(
                base_mvp,
                all_v_pos,
                s=0.9
            )
        else:
            revised_mvp = base_mvp

    if config.cam_jitter_pixels > 0.0:
        pixel_size = 2.0 / float(config.train_res)
        jitter_range = float(config.cam_jitter_pixels) * pixel_size

        jitter_x = (
            torch.rand(batch_size, dtype=dtype, device=device) * 2.0 - 1.0
        ) * jitter_range

        jitter_y = (
            torch.rand(batch_size, dtype=dtype, device=device) * 2.0 - 1.0
        ) * jitter_range

        jitter_matrix = translate_torch(
            jitter_x,
            jitter_y,
            torch.zeros(batch_size, dtype=dtype, device=device),
            batch_size=batch_size,
            device=device,
            dtype=dtype
        )

        revised_mvp = torch.matmul(
            jitter_matrix,
            revised_mvp
        )

    return revised_mvp, campos, lightpos


# ======================================================================================
# Public camera API
# ======================================================================================
def get_camera_params(
    config: DiffRenderConfig,
    opt_mesh: mesh.Mesh,
    ref_mesh: mesh.Mesh,
    mode: str = "train_orbit"
):
    """
    获取当前渲染所需的相机参数 (MVP矩阵, 相机位置, 光源位置)

    mode:
      - train_focus
      - train_orbit
      - preview_fixed
      - preview_orbit
      - preview_raw_focus
      - preview_raw_orbit

    Commit 4 + 5:
      - train_* 模式走 Torch/GPU path，避免每 iteration GPU->CPU 同步。
      - preview_* 模式保留 legacy NumPy path，方便 debug/preview，并降低回归风险。
    """
    if mode.startswith("train"):
        return get_camera_params_torch(
            config=config,
            opt_mesh=opt_mesh,
            ref_mesh=ref_mesh,
            mode=mode
        )

    # -------------------------------------------------------------------------
    # Legacy NumPy implementation for preview/debug paths.
    # -------------------------------------------------------------------------
    is_preview = mode.startswith("preview")
    batch_size = 1 if is_preview else config.batch

    use_training_camera = mode.startswith("train") or mode.startswith("preview_raw")
    use_focus_mode = "focus" in mode

    mvp = np.zeros((batch_size, 4, 4), dtype=np.float32)
    campos = np.zeros((batch_size, 3), dtype=np.float32)
    lightpos = np.zeros((batch_size, 3), dtype=np.float32)

    # 🌟 修复 1：修复透视过度(鱼眼畸变)与近平面裁剪问题
    # 保持 x/n = 0.5 (对应 FOV 约 53 度)，同时将近平面 n 缩小至 0.01
    proj_mtx = util.projection(x=0.005, n=0.01, f=1000.0)
    radius = config.camera_radius

    bmesh_v_pos_used = get_used_vertices(opt_mesh)
    rmesh_v_pos_used = get_used_vertices(ref_mesh)
    all_v_pos = np.concatenate((bmesh_v_pos_used, rmesh_v_pos_used))

    if use_focus_mode:
        # ---------------------------------------------------------
        # 🌟 聚焦模式 (Surface-based Micro-Macro Sampling)
        # ---------------------------------------------------------
        v_pos = opt_mesh.v_pos
        t_pos_idx = opt_mesh.t_pos_idx

        if t_pos_idx.shape[0] == 0:
            targets = torch.zeros((batch_size, 3), device=v_pos.device)
            normals = torch.tensor([0.0, 0.0, 1.0], device=v_pos.device).expand(batch_size, 3)
        else:
            v0 = v_pos[t_pos_idx[:, 0]]
            v1 = v_pos[t_pos_idx[:, 1]]
            v2 = v_pos[t_pos_idx[:, 2]]

            e1 = v1 - v0
            e2 = v2 - v0
            cross = torch.cross(e1, e2, dim=-1)
            area = 0.5 * torch.norm(cross, dim=-1)

            area_sum = area.sum()
            if area_sum <= 1e-6:
                probs = torch.ones_like(area) / area.shape[0]
            else:
                probs = area / area_sum

            sampled_idx = torch.multinomial(probs, batch_size, replacement=True)

            v0_s = v0[sampled_idx]
            v1_s = v1[sampled_idx]
            v2_s = v2[sampled_idx]
            cross_s = cross[sampled_idx]

            normals = torch.nn.functional.normalize(cross_s, dim=-1)

            r1 = torch.rand(batch_size, 1, device=v_pos.device)
            r2 = torch.rand(batch_size, 1, device=v_pos.device)
            sqrt_r1 = torch.sqrt(r1)
            u = 1.0 - sqrt_r1
            v = sqrt_r1 * (1.0 - r2)
            w = sqrt_r1 * r2

            targets = u * v0_s + v * v1_s + w * v2_s

        fluct_in = config.cam_focus_fluctuation_in
        fluct_out = config.cam_focus_fluctuation_out

        rand_dist = torch.rand(batch_size, 1, device=v_pos.device) * (fluct_in + fluct_out) + (1.0 - fluct_in)

        # 🌟 修复 2：为相机距离增加安全下限 (min=0.05)
        # 确保相机距离表面至少有 0.05，完美避开 0.01 的近平面，防止 ref_mesh 穿模被裁剪
        D = torch.clamp(config.cam_focus_offset * rand_dist, min=0.05)

        eyes = targets + normals * D

        targets_np = targets.detach().cpu().numpy()
        eyes_np = eyes.detach().cpu().numpy()
        normals_np = normals.detach().cpu().numpy()
        up_np = np.array(config.camera_up)

        for b in range(batch_size):
            eye_b = eyes_np[b]
            target_b = targets_np[b]
            normal_b = normals_np[b]

            if np.abs(np.dot(normal_b, up_np)) > 0.999:
                up_b = up_np + np.array([0.001, 0.0, 0.0])
                up_b = up_b / np.linalg.norm(up_b)
            else:
                up_b = up_np

            a_mv = util.lookAt(eye_b, target_b, up_b)
            base_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)

            # 智能全景/特写判定
            all_v_homo = np.concatenate([all_v_pos, np.ones((all_v_pos.shape[0], 1), dtype=np.float32)], axis=1)

            # 严格对齐 util.py 的投影矩阵约定 (MVP @ V.T).T
            clip_pos = np.matmul(base_mvp, all_v_homo.T).T
            w = clip_pos[:, 3]

            valid_mask = w > 1e-4
            if np.any(valid_mask):
                valid_clip = clip_pos[valid_mask]
                valid_w = w[valid_mask]
                ndc_pos = valid_clip[:, :3] / valid_w[:, None]
                x = ndc_pos[:, 0]
                y = ndc_pos[:, 1]

                if np.all((x >= -1.0) & (x <= 1.0) & (y >= -1.0) & (y <= 1.0)):
                    # 🌟 接入 enable_auto_center 控制
                    if getattr(config, 'enable_auto_center', True):
                        min_xy = np.min(ndc_pos[:, :2], axis=0)
                        max_xy = np.max(ndc_pos[:, :2], axis=0)
                        center = (min_xy + max_xy) / 2.0

                        size = np.max(max_xy - min_xy) / 2.0
                        if size < 1e-5:
                            size = 1e-5
                        scale = 0.9 / size

                        # 屏幕空间变换必须是左乘 (Scale @ Translate @ MVP)
                        additional_m = np.matmul(util.scale_xy(scale), util.translate(-center[0], -center[1], 0))
                        revised_mvp = np.matmul(additional_m, base_mvp).astype(np.float32)
                    else:
                        revised_mvp = base_mvp
                else:
                    revised_mvp = base_mvp
            else:
                revised_mvp = base_mvp

            # 亚像素抖动
            if config.cam_jitter_pixels > 0.0:
                pixel_size = 2.0 / config.train_res
                jitter_range = config.cam_jitter_pixels * pixel_size
                jitter_x = np.random.uniform(-jitter_range, jitter_range)
                jitter_y = np.random.uniform(-jitter_range, jitter_range)
                jitter_matrix = util.translate(jitter_x, jitter_y, 0)
                # 抖动矩阵左乘
                revised_mvp = np.matmul(jitter_matrix, revised_mvp).astype(np.float32)

            mvp[b] = revised_mvp
            campos[b] = eye_b
            lightpos[b] = util.cosine_sample(eye_b) * radius

        return mvp, campos, lightpos

    else:
        # ---------------------------------------------------------
        # 🌟 轨道相机模式 (Orbit Camera)
        # ---------------------------------------------------------
        for b in range(batch_size):
            if mode == "preview_fixed":
                eye = np.array(config.camera_eye)
                up = np.array(config.camera_up)
                at = np.array([0, 0, 0])
                a_mv = util.lookAt(eye, at, up)
            else:
                r_rot = util.random_rotation_translation(0.25)
                a_mv = np.matmul(util.translate(0, 0, -radius), r_rot)

            base_mvp = np.matmul(proj_mtx, a_mv).astype(np.float32)
            c_pos = np.linalg.inv(a_mv)[:3, 3]

            if not use_training_camera:
                # 预览模式 (fixed 或 orbit)，保持 0.9 缩放看全貌，不加抖动
                # 🌟 核心修复：强制 preview_fixed 始终居中，无视 enable_auto_center 配置
                if mode == "preview_fixed":
                    revised_mvp, _ = util.center_by_screen(base_mvp, all_v_pos, s=0.9, debug_log=False)
                elif getattr(config, 'enable_auto_center', True):
                    revised_mvp, _ = util.center_by_screen(base_mvp, all_v_pos, s=0.9, debug_log=False)
                else:
                    revised_mvp = base_mvp
            else:
                # 训练模式 (train_orbit 或 preview_raw_orbit)，引入“随机特写”与“抖动”数据增强
                if config.cam_zoom_max > 0.0:
                    zoom = np.random.uniform(1.0, 1.0 + config.cam_zoom_max)
                    s_factor = 0.9 * zoom
                    if getattr(config, 'enable_auto_center', True):
                        revised_mvp, _ = util.center_by_screen(base_mvp, all_v_pos, s=s_factor, debug_log=False)
                    else:
                        scaling_matrix = util.scale_xy(s_factor)
                        revised_mvp = np.matmul(scaling_matrix, base_mvp).astype(np.float32)

                    pan_range = (zoom - 1.0) * 0.5
                    pan_x = np.random.uniform(-pan_range, pan_range)
                    pan_y = np.random.uniform(-pan_range, pan_range)
                    pan_matrix = util.translate(pan_x, pan_y, 0)
                    # 平移矩阵左乘
                    revised_mvp = np.matmul(pan_matrix, revised_mvp).astype(np.float32)
                else:
                    if getattr(config, 'enable_auto_center', True):
                        revised_mvp, _ = util.center_by_screen(base_mvp, all_v_pos, s=0.9, debug_log=False)
                    else:
                        revised_mvp = base_mvp

                # Jitter
                if config.cam_jitter_pixels > 0.0:
                    pixel_size = 2.0 / config.train_res
                    jitter_range = config.cam_jitter_pixels * pixel_size
                    jitter_x = np.random.uniform(-jitter_range, jitter_range)
                    jitter_y = np.random.uniform(-jitter_range, jitter_range)
                    jitter_matrix = util.translate(jitter_x, jitter_y, 0)
                    # 抖动矩阵左乘
                    revised_mvp = np.matmul(jitter_matrix, revised_mvp).astype(np.float32)

            mvp[b] = revised_mvp
            campos[b] = c_pos
            lightpos[b] = util.cosine_sample(c_pos) * radius

        return mvp, campos, lightpos
