# filepath: src/ogd_lbs/diff_render/nvdiffmodeling/mesh.py
import os
import numpy as np
import torch

from . import util
from . import texture


######################################################################################
# Base mesh class
######################################################################################
class Mesh:
    def __init__(
        self,
        v_pos=None,
        t_pos_idx=None,
        v_nrm=None,
        t_nrm_idx=None,
        v_tex=None,
        t_tex_idx=None,
        v_tng=None,
        t_tng_idx=None,
        v_weights=None,
        bone_mtx=None,
        material=None,
        v_alpha=None,
        t_alpha_idx=None,
        t_bsdf_id=None,  # 🌟 新增：每个三角面的 BSDF 标签 [F, 1]
        base=None
    ):
        self.v_pos = v_pos
        self.v_weights = v_weights
        self.v_nrm = v_nrm
        self.v_tex = v_tex
        self.v_tng = v_tng

        self.t_pos_idx = t_pos_idx
        self.t_nrm_idx = t_nrm_idx
        self.t_tex_idx = t_tex_idx
        self.t_tng_idx = t_tng_idx

        self.material = material
        self.bone_mtx = bone_mtx

        self.v_color = None
        self.t_color_idx = None

        # 顶点 alpha 属性
        self.v_alpha = v_alpha
        self.t_alpha_idx = t_alpha_idx

        # 🌟 混合材质标签
        self.t_bsdf_id = t_bsdf_id

        if base is not None:
            self.copy_none(base)

    def copy_none(self, other):
        if self.v_pos is None:
            self.v_pos = other.v_pos
        if self.v_weights is None:
            self.v_weights = other.v_weights
        if self.t_pos_idx is None:
            self.t_pos_idx = other.t_pos_idx
        if self.v_nrm is None:
            self.v_nrm = other.v_nrm
        if self.t_nrm_idx is None:
            self.t_nrm_idx = other.t_nrm_idx
        if self.v_tex is None:
            self.v_tex = other.v_tex
        if self.t_tex_idx is None:
            self.t_tex_idx = other.t_tex_idx
        if self.v_tng is None:
            self.v_tng = other.v_tng
        if self.t_tng_idx is None:
            self.t_tng_idx = other.t_tng_idx
        if self.material is None:
            self.material = other.material
        if self.bone_mtx is None:
            self.bone_mtx = other.bone_mtx

        if self.v_color is None:
            self.v_color = other.v_color
        if self.t_color_idx is None:
            self.t_color_idx = other.t_color_idx

        if self.v_alpha is None:
            self.v_alpha = other.v_alpha
        if self.t_alpha_idx is None:
            self.t_alpha_idx = other.t_alpha_idx

        if self.t_bsdf_id is None:
            self.t_bsdf_id = other.t_bsdf_id

    def get_frames(self):
        return self.bone_mtx.shape[0] if self.bone_mtx is not None else 1

    def clone(self):
        out = Mesh(base=self)

        if out.v_pos is not None:
            out.v_pos = out.v_pos.clone()
        if out.v_weights is not None:
            out.v_weights = out.v_weights.clone()
        if out.t_pos_idx is not None:
            out.t_pos_idx = out.t_pos_idx.clone()
        if out.v_nrm is not None:
            out.v_nrm = out.v_nrm.clone()
        if out.t_nrm_idx is not None:
            out.t_nrm_idx = out.t_nrm_idx.clone()
        if out.v_tex is not None:
            out.v_tex = out.v_tex.clone()
        if out.t_tex_idx is not None:
            out.t_tex_idx = out.t_tex_idx.clone()
        if out.v_tng is not None:
            out.v_tng = out.v_tng.clone()
        if out.t_tng_idx is not None:
            out.t_tng_idx = out.t_tng_idx.clone()
        if out.bone_mtx is not None:
            out.bone_mtx = out.bone_mtx.clone()
        if out.v_alpha is not None:
            out.v_alpha = out.v_alpha.clone()
        if out.t_alpha_idx is not None:
            out.t_alpha_idx = out.t_alpha_idx.clone()
        if out.t_bsdf_id is not None:
            out.t_bsdf_id = out.t_bsdf_id.clone()

        return out

    def eval(self, params={}):
        return self


######################################################################################
# Compute AABB
######################################################################################
def aabb(mesh):
    return torch.min(mesh.v_pos, dim=0).values, torch.max(mesh.v_pos, dim=0).values


######################################################################################
# Align base mesh to reference mesh: move & rescale to match bounding boxes.
######################################################################################
def unit_size(mesh):
    with torch.no_grad():
        vmin, vmax = aabb(mesh)
        scale = 2 / torch.max(vmax - vmin).item()
        v_pos = mesh.v_pos - (vmax + vmin) / 2
        v_pos = v_pos * scale

        return Mesh(v_pos, base=mesh)


######################################################################################
# Center & scale mesh for rendering
######################################################################################
def center_by_reference(base_mesh, ref_aabb, scale):
    center = (ref_aabb[0] + ref_aabb[1]) * 0.5
    scale = scale / torch.max(ref_aabb[1] - ref_aabb[0]).item()
    v_pos = (base_mesh.v_pos - center[None, ...]) * scale
    return Mesh(v_pos, base=base_mesh)


######################################################################################
# Rescale base-mesh from NDC [-1, 1] space to same dimensions as reference mesh
######################################################################################
def align_with_reference(base_mesh, ref_mesh):
    class mesh_op_align:
        def __init__(self, base_mesh, ref_mesh):
            self.base_mesh = base_mesh
            with torch.no_grad():
                b_vmin, b_vmax = aabb(base_mesh.eval())
                r_vmin, r_vmax = aabb(ref_mesh.eval())
                b_size = (b_vmax - b_vmin)
                self.offset = (r_vmax + r_vmin) / 2
                self.scale = (r_vmax - r_vmin) / torch.where(
                    b_size > 1e-6,
                    b_size,
                    torch.ones_like(b_size)
                )

        def eval(self, params={}):
            base_mesh = self.base_mesh.eval(params)
            v_pos = base_mesh.v_pos * self.scale[None, ...] + self.offset[None, ...]
            return Mesh(v_pos, base=base_mesh)

    return mesh_op_align(base_mesh, ref_mesh)


######################################################################################
# Skinning
######################################################################################
def _skin_hvec(bone_mtx, weights, attr):
    """
    Optimized linear blend skinning helper.

    Parameters
    ----------
    bone_mtx : torch.Tensor
        Bone transform matrices with shape [B, 4, 4].

    weights : torch.Tensor
        Per-vertex bone weights with shape [V, B].

    attr : torch.Tensor
        Homogeneous vertex attribute with shape [V, 4].
        For positions, w = 1.
        For normals/tangents, w = 0.

    Returns
    -------
    torch.Tensor
        Skinned xyz attribute with shape [V, 3].

    Original semantics
    ------------------
    The old implementation computed:

        out[v] = sum_b weights[v, b] * (attr[v] @ bone_mtx[b])

    using a broadcasted intermediate of shape approximately [B, V, 4].

    This implementation preserves the same row-vector convention by first
    computing the blended matrix for each vertex:

        blended[v] = sum_b weights[v, b] * bone_mtx[b]

    then applying:

        out[v] = attr[v] @ blended[v]

    This reduces the large temporary from O(B * V * 4) to O(V * 16).
    """
    num_bones = bone_mtx.shape[0]
    num_attr = attr.shape[0]

    blended_mtx = torch.matmul(
        weights,
        bone_mtx.reshape(num_bones, 16)
    ).reshape(num_attr, 4, 4)

    attr_out = torch.bmm(
        attr[:, None, :],
        blended_mtx
    ).squeeze(1)

    return attr_out[:, :3]


def skinning(mesh):
    class mesh_op_skinning:
        def __init__(self, input):
            self.input = input

            mesh_eval = self.input.eval()
            t_pos_idx = mesh_eval.t_pos_idx.detach().cpu().numpy()

            if mesh_eval.t_nrm_idx is not None:
                self.nrm_remap = self._compute_remap(
                    t_pos_idx,
                    mesh_eval.v_nrm.shape[0],
                    mesh_eval.t_nrm_idx.detach().cpu().numpy()
                )

            if mesh_eval.t_tng_idx is not None:
                self.tng_remap = self._compute_remap(
                    t_pos_idx,
                    mesh_eval.v_tng.shape[0],
                    mesh_eval.t_tng_idx.detach().cpu().numpy()
                )

        def _compute_remap(self, t_pos_idx, n_attrs, t_attr_idx):
            assert len(t_pos_idx) == len(t_attr_idx)

            attr_vtx_idx = [None] * n_attrs
            for ti in range(0, len(t_pos_idx)):
                for vi in range(0, 3):
                    assert (
                        attr_vtx_idx[t_attr_idx[ti][vi]] is None
                        or attr_vtx_idx[t_attr_idx[ti][vi]] == t_pos_idx[ti][vi]
                    ), "Trying to skin a mesh with shared normals/tangents using different weights"
                    attr_vtx_idx[t_attr_idx[ti][vi]] = t_pos_idx[ti][vi]

            for i, value in enumerate(attr_vtx_idx):
                if value is not None:
                    continue
                attr_vtx_idx[i] = 0

            return torch.tensor(attr_vtx_idx, dtype=torch.int64, device='cuda')

        def eval(self, params={}):
            imesh = self.input.eval(params)

            if imesh.v_weights is None or imesh.bone_mtx is None:
                return imesh

            if torch.is_anomaly_enabled():
                assert torch.all(torch.isfinite(imesh.v_pos))
                assert torch.all(torch.isfinite(imesh.v_weights))

            t_idx = params['time'] if 'time' in params else 0
            t_idx = (t_idx % imesh.bone_mtx.shape[0])

            bone_mtx = imesh.bone_mtx[t_idx, ...]
            bone_mtx_it = torch.transpose(torch.inverse(bone_mtx), -2, -1)

            weights = imesh.v_weights
            assert weights.shape[1] == bone_mtx.shape[0]

            epsilon = 1e-6
            weights = torch.abs(weights)
            weights = weights / (torch.sum(weights, dim=1, keepdim=True) + epsilon)

            weights = weights * params["weight_mask"] if "weight_mask" in params else weights

            v_pos_out = _skin_hvec(
                bone_mtx,
                weights,
                util.to_hvec(imesh.v_pos, 1)
            )

            v_nrm_out = None
            if imesh.v_nrm is not None:
                v_nrm_out = _skin_hvec(
                    bone_mtx_it,
                    weights[self.nrm_remap, ...],
                    util.to_hvec(imesh.v_nrm, 0)
                )
                v_nrm_out = util.safe_normalize(v_nrm_out)

            v_tng_out = None
            if imesh.v_tng is not None:
                v_tng_out = _skin_hvec(
                    bone_mtx,
                    weights[self.tng_remap, ...],
                    util.to_hvec(imesh.v_tng, 0)
                )
                v_tng_out = util.safe_normalize(v_tng_out)

            if torch.is_anomaly_enabled():
                assert torch.all(torch.isfinite(imesh.v_pos))
                assert torch.all(torch.isfinite(weights))
                assert torch.all(torch.isfinite(v_pos_out))
                assert v_nrm_out is None or torch.all(torch.isfinite(v_nrm_out))
                assert v_tng_out is None or torch.all(torch.isfinite(v_tng_out))

            return Mesh(
                v_pos=v_pos_out[:, :3],
                v_nrm=v_nrm_out,
                v_tng=v_tng_out,
                v_alpha=imesh.v_alpha,
                t_alpha_idx=imesh.t_alpha_idx,
                t_bsdf_id=imesh.t_bsdf_id,
                base=imesh
            )

    return mesh_op_skinning(mesh)


def guess_weights(base_mesh, ref_mesh, N=10):
    base_v_pos = base_mesh.v_pos.detach().cpu().numpy()
    ref_v_pos = ref_mesh.v_pos.detach().cpu().numpy()
    ref_v_weights = ref_mesh.v_weights.detach().cpu().numpy()
    base_v_weights = np.zeros((base_v_pos.shape[0], ref_v_weights.shape[1]), dtype=np.float32)

    for v_idx, vtx in enumerate(base_v_pos):
        diff = ref_v_pos - vtx[None, ...]
        dist = np.sum(diff * diff, axis=-1)
        idxs = np.argpartition(dist, N)

        sum_w = 0.0
        sum_vtx_w = np.zeros_like(ref_v_weights[0, ...])
        for i in idxs[:N]:
            sum_w += 1.0 / max(dist[i], 0.001)
            sum_vtx_w += ref_v_weights[i, ...] / max(dist[i], 0.001)
        base_v_weights[v_idx, ...] = sum_vtx_w / sum_w

    return base_v_weights


def random_weights(base_mesh, ref_mesh):
    init = np.random.uniform(
        size=(base_mesh.v_pos.shape[0], ref_mesh.v_weights.shape[1]),
        low=0.0,
        high=1.0
    )
    return init / np.sum(init, axis=1, keepdims=True)


######################################################################################
# Simple smooth vertex normal computation
######################################################################################
def auto_normals(mesh):
    class mesh_op_auto_normals:
        def __init__(self, input):
            self.input = input

        def eval(self, params={}):
            imesh = self.input.eval(params)

            i0 = imesh.t_pos_idx[:, 0]
            i1 = imesh.t_pos_idx[:, 1]
            i2 = imesh.t_pos_idx[:, 2]

            v0 = imesh.v_pos[i0, :]
            v1 = imesh.v_pos[i1, :]
            v2 = imesh.v_pos[i2, :]

            face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)

            v_nrm = torch.zeros_like(imesh.v_pos)
            v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
            v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
            v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

            v_nrm = torch.where(
                util.dot(v_nrm, v_nrm) > 1e-20,
                v_nrm,
                torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')
            )

            self.v_nrm = util.safe_normalize(v_nrm)

            if torch.is_anomaly_enabled():
                assert torch.all(torch.isfinite(self.v_nrm))

            return Mesh(
                v_nrm=self.v_nrm,
                t_nrm_idx=imesh.t_pos_idx,
                base=imesh
            )

    return mesh_op_auto_normals(mesh)


######################################################################################
# Compute tangent space from texture map coordinates
######################################################################################
def compute_tangents(mesh):
    class mesh_op_compute_tangents:
        def __init__(self, input):
            self.input = input

        def eval(self, params={}):
            imesh = self.input.eval(params)

            vn_idx = [None] * 3
            pos = [None] * 3
            tex = [None] * 3

            for i in range(0, 3):
                pos[i] = imesh.v_pos[imesh.t_pos_idx[:, i]]
                tex[i] = imesh.v_tex[imesh.t_tex_idx[:, i]]
                vn_idx[i] = imesh.t_nrm_idx[:, i]

            tangents = torch.zeros_like(imesh.v_nrm)
            tansum = torch.zeros_like(imesh.v_nrm)

            uve1 = tex[1] - tex[0]
            uve2 = tex[2] - tex[0]
            pe1 = pos[1] - pos[0]
            pe2 = pos[2] - pos[0]

            nom = (pe1 * uve2[..., 1:2] - pe2 * uve1[..., 1:2])
            denom = (
                uve1[..., 0:1] * uve2[..., 1:2]
                - uve1[..., 1:2] * uve2[..., 0:1]
            )

            nom = torch.where(
                util.dot(nom, nom) > 1e-20,
                nom,
                torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')
            )

            tang = nom / torch.where(
                denom > 0.0,
                torch.clamp(denom, min=1e-6),
                torch.clamp(denom, max=-1e-6)
            )

            for i in range(0, 3):
                idx = vn_idx[i][:, None].repeat(1, 3)
                tangents.scatter_add_(0, idx, tang)
                tansum.scatter_add_(0, idx, torch.ones_like(tang))

            tangents = torch.where(
                util.dot(tangents, tangents) > 1e-20,
                tangents,
                torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device='cuda')
            )

            tansum = torch.where(
                util.dot(tansum, tansum) > 1e-20,
                tansum,
                torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device='cuda')
            )

            tangents = tangents / tansum

            if torch.is_anomaly_enabled():
                assert torch.all(torch.isfinite(tangents))

            tangents = util.safe_normalize(tangents)
            tangents = util.safe_normalize(
                tangents - util.dot(tangents, imesh.v_nrm) * imesh.v_nrm
            )

            self.v_tng = tangents

            if torch.is_anomaly_enabled():
                assert torch.all(torch.isfinite(tangents))

            return Mesh(
                v_tng=self.v_tng,
                t_tng_idx=imesh.t_nrm_idx,
                base=imesh
            )

    return mesh_op_compute_tangents(mesh)


######################################################################################
# Subdivide each triangle into 4 new ones. Edge midpoint subdivision
######################################################################################
def subdivide(mesh, steps=1):
    class mesh_op_subdivide:
        def __init__(self, input):
            self.input = input
            self.new_vtx_idx = [None] * 4
            self.new_tri_idx = [None] * 4

            imesh = self.input.eval()
            v_attr = [imesh.v_pos, imesh.v_nrm, imesh.v_tex, imesh.v_tng]
            v_idx = [imesh.t_pos_idx, imesh.t_nrm_idx, imesh.t_tex_idx, imesh.t_tng_idx]

            for i, attr in enumerate(v_attr):
                if attr is not None:
                    tri_idx = v_idx[i].cpu().numpy()

                    edge_fetch_a = []
                    edge_fetch_b = []
                    edge_verts = {}

                    for tri in tri_idx:
                        for e_idx in range(0, 3):
                            v0 = tri[e_idx]
                            v1 = tri[(e_idx + 1) % 3]
                            if (v1, v0) not in edge_verts.keys():
                                edge_verts[(v0, v1)] = [len(edge_fetch_a), v0, v1]
                                edge_fetch_a += [v0]
                                edge_fetch_b += [v1]

                    self.new_vtx_idx[i] = [
                        torch.tensor(edge_fetch_a, dtype=torch.int64, device='cuda'),
                        torch.tensor(edge_fetch_b, dtype=torch.int64, device='cuda')
                    ]

                    new_tri_idx = []
                    for tri in tri_idx:
                        v0, v1, v2 = tri
                        h0 = (
                            edge_verts[(v0, v1)][0]
                            if (v0, v1) in edge_verts.keys()
                            else edge_verts[(v1, v0)][0]
                        ) + attr.shape[0]
                        h1 = (
                            edge_verts[(v1, v2)][0]
                            if (v1, v2) in edge_verts.keys()
                            else edge_verts[(v2, v1)][0]
                        ) + attr.shape[0]
                        h2 = (
                            edge_verts[(v2, v0)][0]
                            if (v2, v0) in edge_verts.keys()
                            else edge_verts[(v0, v2)][0]
                        ) + attr.shape[0]

                        new_tri_idx += [
                            [v0, h0, h2],
                            [h0, v1, h1],
                            [h1, v2, h2],
                            [h0, h1, h2]
                        ]

                    self.new_tri_idx[i] = torch.tensor(
                        new_tri_idx,
                        dtype=torch.int64,
                        device='cuda'
                    )

        def eval(self, params={}):
            imesh = self.input.eval(params)

            v_attr = [imesh.v_pos, imesh.v_nrm, imesh.v_tex, imesh.v_tng]
            v_idx = [imesh.t_pos_idx, imesh.t_nrm_idx, imesh.t_tex_idx, imesh.t_tng_idx]

            for i, attr in enumerate(v_attr):
                if attr is not None:
                    edge_attr = (
                        attr[self.new_vtx_idx[i][0], :]
                        + attr[self.new_vtx_idx[i][1], :]
                    ) * 0.5

                    v_attr[i] = torch.cat([attr, edge_attr], dim=0)
                    v_idx[i] = self.new_tri_idx[i]

            # 🌟 修复：同步扩充 t_bsdf_id，确保细分后标签与面片数量一致
            new_t_bsdf_id = None
            if imesh.t_bsdf_id is not None:
                new_t_bsdf_id = imesh.t_bsdf_id.repeat_interleave(4, dim=0)

            return Mesh(
                v_pos=v_attr[0],
                t_pos_idx=v_idx[0],
                v_nrm=v_attr[1],
                t_nrm_idx=v_idx[1],
                v_tex=v_attr[2],
                t_tex_idx=v_idx[2],
                v_tng=v_attr[3],
                t_tng_idx=v_idx[3],
                t_bsdf_id=new_t_bsdf_id,
                base=imesh
            )

    x = mesh
    for _ in range(steps):
        x = mesh_op_subdivide(x)

    bm = mesh.eval()
    sm = x.eval()

    v_attr_orig = [bm.v_pos, bm.v_nrm, bm.v_tex, bm.v_tng]
    v_attr = [sm.v_pos, sm.v_nrm, sm.v_tex, sm.v_tng]
    v_idx_orig = [bm.t_pos_idx, bm.t_nrm_idx, bm.t_tex_idx, bm.t_tng_idx]
    v_idx = [sm.t_pos_idx, sm.t_nrm_idx, sm.t_tex_idx, sm.t_tng_idx]

    print("Subdivided mesh:")
    print(
        "    Attrs:   [%6d, %6d, %6d, %6d] -> [%6d, %6d, %6d, %6d]"
        % tuple(
            list((a.shape[0] if a is not None else 0) for a in v_attr_orig)
            + list((a.shape[0] if a is not None else 0) for a in v_attr)
        )
    )
    print(
        "    Indices: [%6d, %6d, %6d, %6d] -> [%6d, %6d, %6d, %6d]"
        % tuple(
            list((a.shape[0] if a is not None else 0) for a in v_idx_orig)
            + list((a.shape[0] if a is not None else 0) for a in v_idx)
        )
    )

    return x


######################################################################################
# Displacement mapping
######################################################################################
def displace(mesh, displacement_map, scale=1.0, keep_connectivity=True):
    class mesh_op_displace:
        def __init__(self, input, displacement_map, scale, keep_connectivity):
            self.input = input
            self.displacement_map = displacement_map
            self.scale = scale
            self.keep_connectivity = keep_connectivity

        def eval(self, params={}):
            imesh = self.input.eval(params)

            if self.keep_connectivity:
                vd = torch.zeros_like(imesh.v_pos)
                vd_n = torch.zeros_like(imesh.v_pos)

                for i in range(0, 3):
                    v = imesh.v_pos[imesh.t_pos_idx[:, i], :]
                    n = imesh.v_nrm[imesh.t_nrm_idx[:, i], :]
                    t = imesh.v_tex[imesh.t_tex_idx[:, i], :]
                    v_displ = v + n * self.scale * util.tex_2d(self.displacement_map, t)

                    splat_idx = imesh.t_pos_idx[:, i, None].repeat(1, 3)
                    vd.scatter_add_(0, splat_idx, v_displ)
                    vd_n.scatter_add_(0, splat_idx, torch.ones_like(v_displ))

                return Mesh(vd / vd_n, base=imesh)

            vd = torch.zeros(
                [imesh.v_tex.shape[0], 3],
                dtype=torch.float32,
                device='cuda'
            )
            vd_n = torch.zeros(
                [imesh.v_tex.shape[0], 3],
                dtype=torch.float32,
                device='cuda'
            )

            for i in range(0, 3):
                v = imesh.v_pos[imesh.t_pos_idx[:, i], :]
                n = imesh.v_nrm[imesh.t_nrm_idx[:, i], :]
                t = imesh.v_tex[imesh.t_tex_idx[:, i], :]
                v_displ = v + n * self.scale * util.tex_2d(self.displacement_map, t)

                splat_idx = imesh.t_tex_idx[:, i, None].repeat(1, 3)
                vd.scatter_add_(0, splat_idx, v_displ)
                vd_n.scatter_add_(0, splat_idx, torch.ones_like(v_displ))

            return Mesh(vd / vd_n, mesh.t_tex_idx, base=imesh)

    return mesh_op_displace(mesh, displacement_map, scale, keep_connectivity)


######################################################################################
# Utilities to merge meshes / materials. No mesh-ops or differentiable stuff here.
######################################################################################
def merge(mesh_a, mesh_b):
    def _merge_attr_idx(a, b, a_idx, b_idx):
        if a is None and b is None:
            return None, None
        elif a is not None and b is None:
            return a, a_idx
        elif a is None and b is not None:
            return b, b_idx
        else:
            return torch.cat((a, b), dim=0), torch.cat((a_idx, b_idx + a.shape[0]), dim=0)

    v_pos, t_pos_idx = _merge_attr_idx(mesh_a.v_pos, mesh_b.v_pos, mesh_a.t_pos_idx, mesh_b.t_pos_idx)
    v_nrm, t_nrm_idx = _merge_attr_idx(mesh_a.v_nrm, mesh_b.v_nrm, mesh_a.t_nrm_idx, mesh_b.t_nrm_idx)
    v_tng, t_tng_idx = _merge_attr_idx(mesh_a.v_tng, mesh_b.v_tng, mesh_a.t_tng_idx, mesh_b.t_tng_idx)
    v_tex, t_tex_idx = _merge_attr_idx(mesh_a.v_tex, mesh_b.v_tex, mesh_a.t_tex_idx, mesh_b.t_tex_idx)

    if mesh_a.v_weights is None and mesh_b.v_weights is None:
        v_weights, bone_mtx = None, None
    elif mesh_a.v_weights is not None and mesh_b.v_weights is None:
        v_weights, bone_mtx = mesh_a.v_weights, mesh_a.bone_mtx
    elif mesh_a.v_weights is None and mesh_b.v_weights is not None:
        v_weights, bone_mtx = mesh_b.v_weights, mesh_b.bone_mtx
    else:
        if torch.all(mesh_a.bone_mtx == mesh_b.bone_mtx):
            bone_mtx = mesh_a.bone_mtx
            v_weights = torch.cat((mesh_a.v_weights, mesh_b.v_weights), dim=0)
        else:
            bone_mtx = torch.cat((mesh_a.bone_mtx, mesh_b.bone_mtx), dim=1)
            v_wa = torch.nn.functional.pad(mesh_a.v_weights, [0, mesh_b.v_weights.shape[1]])
            v_wb = torch.nn.functional.pad(mesh_b.v_weights, [mesh_a.v_weights.shape[1], 0])
            v_weights = torch.cat((v_wa, v_wb), dim=0)

    return Mesh(
        v_pos=v_pos,
        t_pos_idx=t_pos_idx,
        v_nrm=v_nrm,
        t_nrm_idx=t_nrm_idx,
        v_tng=v_tng,
        t_tng_idx=t_tng_idx,
        v_tex=v_tex,
        t_tex_idx=t_tex_idx,
        v_weights=v_weights,
        bone_mtx=bone_mtx,
        base=mesh_a
    )


def sub_mesh(mesh, sub_t_idx):
    class mesh_op_sub_mesh:
        def __init__(self, input, sub_t_idx):
            self.input = input
            self.sub_t_idx = sub_t_idx

        def eval(self, params={}):
            imesh = self.input.eval(params)

            return Mesh(
                t_pos_idx=imesh.t_pos_idx[sub_t_idx, :],
                t_nrm_idx=(
                    imesh.t_nrm_idx[sub_t_idx, :]
                    if imesh.t_nrm_idx is not None
                    else None
                ),
                t_tex_idx=(
                    imesh.t_tex_idx[sub_t_idx, :]
                    if imesh.t_tex_idx is not None
                    else None
                ),
                t_tng_idx=(
                    imesh.t_tng_idx[sub_t_idx, :]
                    if imesh.t_tng_idx is not None
                    else None
                ),
                t_alpha_idx=(
                    imesh.t_alpha_idx[sub_t_idx, :]
                    if imesh.t_alpha_idx is not None
                    else None
                ),
                t_bsdf_id=(
                    imesh.t_bsdf_id[sub_t_idx, :]
                    if imesh.t_bsdf_id is not None
                    else None
                ),
                base=imesh
            )

    return mesh_op_sub_mesh(mesh, sub_t_idx)


def mtl_override(mesh, mtl_override):
    class mesh_op_mtl_override:
        def __init__(self, input, mtl_override):
            self.input = input
            self.mtl_override = mtl_override

        def eval(self, params={}):
            imesh = self.input.eval(params)

            return Mesh(
                material=mtl_override,
                base=imesh
            )

    return mesh_op_mtl_override(mesh, mtl_override)
