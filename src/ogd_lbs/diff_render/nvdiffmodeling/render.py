# filepath: src/ogd_lbs/diff_render/nvdiffmodeling/render.py
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from . import util
from . import mesh
from . import renderutils as ru


# ==============================================================================================
#  Helper functions
# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr.contiguous(),
        rast,
        attr_idx,
        rast_db=rast_db,
        diff_attrs=None if rast_db is None else 'all'
    )


# ==============================================================================================
#  pixel shader
# ==============================================================================================
def shade(
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        light_pos,
        light_power,
        material,
        min_roughness,
        gb_color,
        gb_alpha=None,
        gb_bsdf_id=None  # 🌟 像素级 BSDF 标签
    ):

    ################################################################################
    # Texture lookups
    ################################################################################

    kd = material['kd'].sample(gb_texc, gb_texc_deriv, "nearest")
    ks = material['ks'].sample(gb_texc, gb_texc_deriv)[..., 0:3]

    perturbed_nrm = None
    if 'normal' in material:
        perturbed_nrm = material['normal'].sample(gb_texc, gb_texc_deriv)

    gb_normal_2sided = ru.prepare_shading_normal(
        gb_pos,
        view_pos,
        perturbed_nrm,
        gb_normal,
        gb_tangent,
        gb_geometric_normal,
        two_sided_shading=True,
        opengl=True
    )

    gb_normal_1sided = ru.prepare_shading_normal(
        gb_pos,
        view_pos,
        perturbed_nrm,
        gb_normal,
        gb_tangent,
        gb_geometric_normal,
        two_sided_shading=False,
        opengl=True
    )

    # Separate kd into alpha and color, default alpha = 1
    alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1])
    kd = kd[..., 0:3]

    if gb_alpha is not None and material.get("use_vertex_alpha", True):
        alpha = alpha * torch.clamp(gb_alpha, 0.0, 1.0)

    ################################################################################
    # Common directions and masks
    ################################################################################

    light_dir = util.safe_normalize(light_pos - gb_pos)
    view_dir = util.safe_normalize(view_pos - gb_pos)

    # front_mask:
    #   True  = 当前像素对应正面
    #   False = 当前像素对应背面
    #
    # 1-sided BSDF 通过 alpha *= front_mask 实现背面剔除。
    front_mask = (
        torch.sum(
            gb_normal_1sided * view_dir,
            -1,
            keepdim=True
        ) > 0
    )

    front_mask_f = front_mask.float()
    front_mask_rgb = front_mask.repeat(1, 1, 1, 3)

    ################################################################################
    # Evaluate BSDF
    ################################################################################

    assert 'bsdf' in material, "Material must specify a BSDF type"

    if material['bsdf'] == 'pbr':
        shaded_col = ru.pbr_bsdf(
            kd,
            ks,
            gb_pos,
            gb_normal,
            view_pos,
            light_pos,
            min_roughness
        ) * light_power

    elif material['bsdf'] == 'diffuse':
        # Legacy diffuse：语义接近 diffuse_2sided。
        shaded_col = kd * (
            ru.lambert(
                gb_normal_2sided,
                light_dir
            ) * 0.7 + 0.3
        ) * light_power

        if material.get('back_deepen', False):
            shaded_col = torch.where(
                front_mask_rgb,
                shaded_col,
                shaded_col * 0.5
            )

    elif material['bsdf'] == 'diffuse_1sided':
        # 🌟 单面 diffuse。
        # 背面 alpha=0，正面保留 diffuse 光照。
        shaded_col = kd * (
            ru.lambert(
                gb_normal_1sided,
                light_dir
            ) * 0.7 + 0.3
        ) * light_power

        if material.get('back_deepen', False):
            shaded_col = torch.where(
                front_mask_rgb,
                shaded_col,
                shaded_col * 0.5
            )

        alpha = alpha * front_mask_f

    elif material['bsdf'] == 'diffuse_2sided':
        # 🌟 双面 diffuse。
        # 语义基本等价于 legacy diffuse，但显式可控，方便 mixed BSDF ID 统计。
        shaded_col = kd * (
            ru.lambert(
                gb_normal_2sided,
                light_dir
            ) * 0.7 + 0.3
        ) * light_power

        if material.get('back_deepen', False):
            shaded_col = torch.where(
                front_mask_rgb,
                shaded_col,
                shaded_col * 0.5
            )

    elif material['bsdf'] == 'diffuse_2sided_back_deepen':
        # 🌟 显式双面 diffuse + 背面加深。
        # 这个 direct BSDF 主要用于调试或非 mixed 模式。
        # two-stage Stage01 的逐子网格规则通常走 material['bsdf'] == 'mixed'
        # 并通过 gb_bsdf_id == 5 进入同样逻辑。
        shaded_col = kd * (
            ru.lambert(
                gb_normal_2sided,
                light_dir
            ) * 0.7 + 0.3
        ) * light_power

        shaded_col = torch.where(
            front_mask_rgb,
            shaded_col,
            shaded_col * 0.5
        )

    # 🌟 混合材质管线
    elif material['bsdf'] == 'mixed':
        assert gb_bsdf_id is not None, "Mixed BSDF requires t_bsdf_id on the mesh."

        # ID 约定：
        #   0: diffuse                         legacy diffuse，语义接近 diffuse_2sided
        #   1: ambient_1sided                  无光照，背面 alpha=0
        #   2: ambient_2sided                  无光照，双面可见
        #   3: diffuse_1sided                  diffuse 光照，背面 alpha=0
        #   4: diffuse_2sided                  diffuse 光照，双面可见
        #   5: diffuse_2sided_back_deepen      diffuse 光照，双面可见，但背面颜色加深
        #
        # two-stage Stage01 几何阶段转换策略：
        #   ambient_1sided -> diffuse_2sided_back_deepen, ID=5
        #   ambient_2sided -> diffuse_2sided,             ID=4
        #
        # 注意：
        #   ID=5 的 back_deepen 是逐面语义，不依赖 material["back_deepen"]。
        #   material["back_deepen"] 只作为 legacy/global 开关，谨慎使用。

        diffuse_2sided_col = kd * (
            ru.lambert(
                gb_normal_2sided,
                light_dir
            ) * 0.7 + 0.3
        ) * light_power

        diffuse_1sided_col = kd * (
            ru.lambert(
                gb_normal_1sided,
                light_dir
            ) * 0.7 + 0.3
        ) * light_power

        diffuse_2sided_deep_col = torch.where(
            front_mask_rgb,
            diffuse_2sided_col,
            diffuse_2sided_col * 0.5
        )

        diffuse_1sided_deep_col = torch.where(
            front_mask_rgb,
            diffuse_1sided_col,
            diffuse_1sided_col * 0.5
        )

        ambient_col = kd * 1.0

        is_diffuse_legacy = gb_bsdf_id == 0
        is_ambient_1sided = gb_bsdf_id == 1
        is_ambient_2sided = gb_bsdf_id == 2
        is_diffuse_1sided = gb_bsdf_id == 3
        is_diffuse_2sided = gb_bsdf_id == 4
        is_diffuse_2sided_back_deepen = gb_bsdf_id == 5

        # 默认 fallback：diffuse_2sided。
        # 这样未知 ID 不会意外变成 ambient，也不会意外剔除 alpha。
        shaded_col = diffuse_2sided_col

        # ambient family
        shaded_col = torch.where(
            is_ambient_1sided | is_ambient_2sided,
            ambient_col,
            shaded_col
        )

        # explicit diffuse_1sided
        shaded_col = torch.where(
            is_diffuse_1sided,
            diffuse_1sided_col,
            shaded_col
        )

        # explicit diffuse_2sided_back_deepen
        # 原始 ambient_1sided 在 Stage01 会被转换为这个 ID。
        # 它是双面可见 diffuse，但背面颜色加深。
        shaded_col = torch.where(
            is_diffuse_2sided_back_deepen,
            diffuse_2sided_deep_col,
            shaded_col
        )

        # legacy/global material["back_deepen"] 支持。
        # 注意：
        #   正常 two-stage Stage01 不应该再依赖这个全局开关，
        #   否则 ambient_2sided -> diffuse_2sided 的面也会被误加深。
        #   这里只保留向后兼容：如果某些旧路径显式设置 material["back_deepen"]=True，
        #   那么 legacy diffuse / diffuse_1sided / diffuse_2sided 也会加深。
        if material.get('back_deepen', False):
            shaded_col = torch.where(
                is_diffuse_legacy | is_diffuse_2sided,
                diffuse_2sided_deep_col,
                shaded_col
            )

            shaded_col = torch.where(
                is_diffuse_1sided,
                diffuse_1sided_deep_col,
                shaded_col
            )

            # ID=5 已经是 deep，不需要额外处理。

        # 1-sided family 统一剔除背面。
        # 注意：ID=5 是 diffuse_2sided_back_deepen，不剔除背面。
        is_one_sided = is_ambient_1sided | is_diffuse_1sided

        alpha = torch.where(
            is_one_sided,
            alpha * front_mask_f,
            alpha
        )

    elif material['bsdf'] == 'normal':
        shaded_col = (gb_normal + 1.0) * 0.5

    elif material['bsdf'] == 'position':
        shaded_col = gb_pos

    elif material['bsdf'] == 'mask':
        shaded_col = torch.ones_like(gb_pos)

    elif material['bsdf'] == 'tangent':
        shaded_col = (gb_tangent + 1.0) * 0.5

    elif material['bsdf'] == 'ambient':
        shaded_col = kd * 1.0

    elif material['bsdf'] == 'ambient_1sided':
        alpha = alpha * front_mask_f
        shaded_col = kd * 1.0

    elif material['bsdf'] == 'ambient_2sided':
        shaded_col = kd * 1.0

    elif material['bsdf'] == 'ambient_4_metric':
        alpha_mask = front_mask
        alpha_mask = (alpha_mask + gb_color[..., 0:1]) > 0
        alpha *= alpha_mask
        shaded_col = kd * 1.0

    elif material['bsdf'] == 'ambient_4_metric_alpha_clip':
        alpha_mask = front_mask
        alpha_mask = (alpha_mask + gb_color[..., 0:1]) > 0
        alpha *= alpha_mask
        alpha = torch.where(
            alpha > 0.5,
            torch.ones_like(alpha),
            torch.zeros_like(alpha)
        )
        shaded_col = kd * 1.0

    else:
        assert False, "Invalid BSDF '%s'" % material['bsdf']

    alpha_clip_threshold = material.get("alpha_clip_threshold", None)
    if alpha_clip_threshold is not None:
        threshold = float(alpha_clip_threshold)
        alpha = torch.where(
            alpha > threshold,
            torch.ones_like(alpha),
            torch.zeros_like(alpha)
        )

    out = torch.cat((shaded_col, alpha), dim=-1)
    return out


# ==============================================================================================
#  Render a depth peeled mesh
# ==============================================================================================
def render_layer(
        rast,
        rast_deriv,
        mesh,
        view_pos,
        light_pos,
        light_power,
        resolution,
        min_roughness,
        spp,
        msaa
    ):

    full_res = resolution * spp

    ################################################################################
    # Rasterize
    ################################################################################

    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(
            rast,
            [resolution, resolution],
            mag='nearest',
            min='nearest'
        )
        rast_out_deriv_s = util.scale_img_nhwc(
            rast_deriv,
            [resolution, resolution],
            mag='nearest',
            min='nearest'
        ) * spp
    else:
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

    ################################################################################
    # Interpolate attributes
    ################################################################################

    gb_pos, _ = interpolate(
        mesh.v_pos[None, ...],
        rast_out_s,
        mesh.t_pos_idx.int()
    )

    v0 = mesh.v_pos[mesh.t_pos_idx[:, 0], :]
    v1 = mesh.v_pos[mesh.t_pos_idx[:, 1], :]
    v2 = mesh.v_pos[mesh.t_pos_idx[:, 2], :]

    face_normals = util.safe_normalize(
        torch.cross(v1 - v0, v2 - v0, dim=-1)
    )

    face_normal_indices = (
        torch.arange(
            0,
            face_normals.shape[0],
            dtype=torch.int64,
            device='cuda'
        )[:, None]
    ).repeat(1, 3)

    gb_geometric_normal, _ = interpolate(
        face_normals[None, ...],
        rast_out_s,
        face_normal_indices.int()
    )

    assert mesh.v_nrm is not None and mesh.v_tng is not None

    gb_normal, _ = interpolate(
        mesh.v_nrm[None, ...],
        rast_out_s,
        mesh.t_nrm_idx.int()
    )

    gb_tangent, _ = interpolate(
        mesh.v_tng[None, ...],
        rast_out_s,
        mesh.t_tng_idx.int()
    )

    if mesh.v_color is not None and mesh.t_color_idx is not None:
        gb_color, _ = interpolate(
            mesh.v_color[None, ...],
            rast_out_s,
            mesh.t_color_idx.int()
        )
    else:
        gb_color = None

    if mesh.v_alpha is not None:
        alpha_idx = mesh.t_alpha_idx if mesh.t_alpha_idx is not None else mesh.t_pos_idx
        gb_alpha, _ = interpolate(
            mesh.v_alpha[None, ...],
            rast_out_s,
            alpha_idx.int()
        )
    else:
        gb_alpha = None

    # 🌟 修复：安全提取像素级 BSDF ID，增加越界保护
    gb_bsdf_id = None
    if mesh.material.get("bsdf", None) == "mixed":
        if mesh.t_bsdf_id is None:
            raise RuntimeError("Mixed BSDF requires mesh.t_bsdf_id, but got None.")

        num_faces = mesh.t_pos_idx.shape[0]
        if mesh.t_bsdf_id.shape[0] != num_faces:
            raise RuntimeError(
                f"t_bsdf_id length mismatch: "
                f"t_bsdf_id={mesh.t_bsdf_id.shape[0]}, faces={num_faces}"
            )

        if num_faces == 0:
            gb_bsdf_id = torch.zeros_like(rast_out_s[..., 3:4].long())
        else:
            tri_id_raw = rast_out_s[..., 3].long() - 1
            covered = tri_id_raw >= 0

            # Clamp 上下界，防止非法索引
            safe_tri_id = torch.where(
                covered,
                torch.clamp(tri_id_raw, min=0, max=num_faces - 1),
                torch.zeros_like(tri_id_raw)
            )

            gb_bsdf_id = mesh.t_bsdf_id[safe_tri_id]

            # 背景像素强制设为 0
            gb_bsdf_id = torch.where(
                covered[..., None],
                gb_bsdf_id,
                torch.zeros_like(gb_bsdf_id)
            )

    assert mesh.v_tex is not None

    gb_texc, gb_texc_deriv = interpolate(
        mesh.v_tex[None, ...],
        rast_out_s,
        mesh.t_tex_idx.int(),
        rast_db=rast_out_deriv_s
    )

    ################################################################################
    # Shade
    ################################################################################

    color = shade(
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        light_pos,
        light_power,
        mesh.material,
        min_roughness,
        gb_color,
        gb_alpha,
        gb_bsdf_id  # 🌟 传入混合材质标签
    )

    ################################################################################
    # Prepare output
    ################################################################################

    if spp > 1 and msaa:
        color = util.scale_img_nhwc(
            color,
            [full_res, full_res],
            mag='nearest',
            min='nearest'
        )

    return color


# ==============================================================================================
#  Render a depth peeled mesh
# ==============================================================================================
def render_mesh(
        ctx,
        mesh,
        mtx_in,
        view_pos,
        light_pos,
        light_power,
        resolution,
        spp                       = 1,
        num_layers                = 1,
        msaa                      = False,
        background                = None,
        antialias                 = True,
        min_roughness             = 0.08,
        return_alpha              = False  # 🌟 新增：支持返回 RGBA
    ):

    def prepare_input_vector(x):
        x = torch.tensor(
            x,
            dtype=torch.float32,
            device='cuda'
        ) if not torch.is_tensor(x) else x

        return x[:, None, None, :] if len(x.shape) == 2 else x

    full_res = resolution * spp

    mtx_in = torch.tensor(
        mtx_in,
        dtype=torch.float32,
        device='cuda'
    ) if not torch.is_tensor(mtx_in) else mtx_in

    # 🌟 修复：空网格拦截保护，防止 nvdiffrast 崩溃
    if mesh.t_pos_idx is None or mesh.t_pos_idx.numel() == 0 or mesh.t_pos_idx.shape[0] == 0:
        if background is not None:
            if spp > 1:
                background = util.scale_img_nhwc(
                    background,
                    [full_res, full_res],
                    mag='nearest',
                    min='nearest'
                )
            out = util.avg_pool_nhwc(background, spp) if spp > 1 else background
            return out if return_alpha and out.shape[-1] == 4 else out[..., :3].contiguous()
        else:
            batch_size = mtx_in.shape[0]
            channels = 4 if return_alpha else 3
            return torch.zeros(
                size=(batch_size, resolution, resolution, channels),
                dtype=torch.float32,
                device='cuda'
            )

    light_pos = prepare_input_vector(light_pos)
    light_power = prepare_input_vector(light_power)
    view_pos = prepare_input_vector(view_pos)

    v_pos_clip = ru.xfm_points(
        mesh.v_pos[None, ...],
        mtx_in,
        use_python=True
    )

    layers = []
    with dr.DepthPeeler(
        ctx,
        v_pos_clip,
        mesh.t_pos_idx.int(),
        [resolution * spp, resolution * spp]
    ) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            layers += [(
                render_layer(
                    rast,
                    db,
                    mesh,
                    view_pos,
                    light_pos,
                    light_power,
                    resolution,
                    min_roughness,
                    spp,
                    msaa
                ),
                rast
            )]

    if background is not None:
        assert background.shape[1] == resolution and background.shape[2] == resolution

        if spp > 1:
            background = util.scale_img_nhwc(
                background,
                [full_res, full_res],
                mag='nearest',
                min='nearest'
            )

        accum_col = background
    else:
        channels = 4 if return_alpha else 3
        accum_col = torch.zeros(
            size=(1, full_res, full_res, channels),
            dtype=torch.float32,
            device='cuda'
        )

    for color, rast in reversed(layers):
        alpha = (rast[..., -1:] > 0) * color[..., 3:4]

        if background is not None:
            accum_col = torch.lerp(accum_col, color[..., 0:3], alpha)
        else:
            if return_alpha:
                # 🌟 正确累加 RGBA
                accum_rgb = torch.lerp(accum_col[..., 0:3], color[..., 0:3], alpha)
                accum_a = alpha + accum_col[..., 3:4] * (1.0 - alpha)
                accum_col = torch.cat([accum_rgb, accum_a], dim=-1)
            else:
                accum_col = torch.lerp(accum_col, color[..., 0:3], alpha)

        if antialias:
            accum_col = dr.antialias(
                accum_col.contiguous(),
                rast,
                v_pos_clip,
                mesh.t_pos_idx.int()
            )

    out = util.avg_pool_nhwc(accum_col, spp) if spp > 1 else accum_col

    if not return_alpha:
        out = out[..., :3]

    return out.contiguous()
