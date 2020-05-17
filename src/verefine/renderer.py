# Author: Dominik Bauer
# Vision for Robotics Group, Automation and Control Institute (ACIN)
# TU Wien, Vienna

import OpenGL
OpenGL.ERROR_CHECKING = True
import OpenGL.GL as gl
from glumpy import gloo, app
app.use('glfw')
import numpy as np
import os

import src.verefine.config as config


class Model:

    _idx_offset = 0  # offset added to indices in index buffer to address joint vertex buffer correctly

    def __init__(self, mesh=None, t_offset=None):

        if mesh is not None:
            # load using trimesh
            pts = mesh.vertices.view(np.ndarray).astype(np.float32)
            if not hasattr(mesh.visual, "uv"):
                texture_uv = np.zeros((pts.shape[0], 2))
            else:
                texture_uv = mesh.visual.uv.view(np.ndarray).astype(np.float32)
            faces = mesh.faces.view(np.ndarray).astype(np.uint32).reshape(-1,)

            # extend vertices to faces size - allows to fix models with broken normal information
            pts = pts[faces]
            normals = np.asarray([[fn, fn, fn] for fn in mesh.face_normals]).reshape(-1, 3)
            texture_uv = texture_uv[faces]
            faces = np.arange(pts.shape[0])

            # prepare buffers -- we use a single VBO, thus indices need to be offset
            vertices_type = [('a_position', np.float32, 3),
                             ('a_normal', np.float32, 3),
                             ('a_texcoord', np.float32, 2)]

            self.vertex_buffer = np.array(list(zip(pts, normals, texture_uv)), vertices_type)
            self.index_buffer = (faces.flatten().astype(np.uint32) + Model._idx_offset).view(gloo.IndexBuffer)

            Model._idx_offset += self.vertex_buffer.shape[0]
        else:
            self.vertex_buffer, self.index_buffer = None, None

        # model space offset
        self.mat_offset = np.eye(4)
        if t_offset is not None:
            self.mat_offset[:3, 3] = t_offset

    def draw(self, program):
        program.draw(gl.GL_TRIANGLES, self.index_buffer)


class ScreenQuad(Model):

    def __init__(self):
        Model.__init__(self)
        pts = np.array([
            [-1, -1, 0],
            [ 1, -1, 0],
            [ 1,  1, 0],
            [-1,  1, 0]
        ]).astype(np.float32)
        normals = np.array([[0, 0, 1]]*4).astype(np.float32)
        uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]]).astype(np.float32)
        faces = np.array([[0, 3, 2], [0, 2, 1]]).astype(np.uint32).reshape(-1)

        vertices_type = [('a_position', np.float32, 3),
                         ('a_normal', np.float32, 3),
                         ('a_texcoord', np.float32, 2)]
        vertices = np.array(list(zip(pts, normals, uvs)), vertices_type)

        # prepare buffers -- we use a single VBO, thus indices need to be offset
        self.vertex_buffer = vertices
        self.index_buffer = (faces.astype(np.uint32) + Model._idx_offset).view(gloo.IndexBuffer)

        Model._idx_offset += vertices.shape[0]


class Renderer:

    def __init__(self, dataset, width=640, height=480):
        self.width, self.height = width, height
        self.near, self.far = config.CLIP_NEAR, config.CLIP_FAR
        self.window = app.Window(width, height, visible=False)

        # FBOs: one per stage
        self.normal_depth_buf = np.zeros((height, width, 4), np.float32).view(gloo.TextureFloat2D)
        self.depth_buf = np.zeros((height, width), np.float32).view(gloo.DepthTexture)
        self.fbo_stage_1 = gloo.FrameBuffer(color=self.normal_depth_buf, depth=self.depth_buf)

        self.cost_buf = np.zeros((height, width, 4), np.float32).view(gloo.TextureFloat2D)
        self.fbo_stage_2 = gloo.FrameBuffer(color=self.cost_buf)

        # load models
        self.dataset = dataset
        self.models = [Model(mesh, offset) for mesh, offset in zip(self.dataset.meshes, self.dataset.obj_model_offset)]
        self.models.append(ScreenQuad())  # used for stage 2

        # VBO: one for all models - bind once, draw according to model indices
        self.vertex_buffer = np.hstack([model.vertex_buffer for model in self.models]).view(gloo.VertexBuffer)
        for model in self.models:
            model.vertex_buffer = self.vertex_buffer

        # shader
        basepath = os.path.dirname(os.path.abspath(__file__))
        with open(f"{basepath}/score.vert", 'r') as file:
            shader_vertex = "".join(file.readlines())
        with open(f"{basepath}/score.frag", 'r') as file:
            shader_fragment = "".join(file.readlines())
        self.program = gloo.Program(shader_vertex, shader_fragment)
        self.program['u_texture'] = np.zeros((512, 512, 3), np.float32)
        self.program.bind(self.vertex_buffer)  # is bound once -- saves some time

        gl.glViewport(0, 0, self.width, self.height)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)  # fixes rendering of single-sided lamp mesh in LINEMOD

        self.observation = None

    def set_observation(self, observation_d, observation_n):
        if self.observation is not None:
            self.observation.delete()
        self.observation = (np.dstack([observation_n, observation_d]).astype(np.float32)[::-1, :, :])\
            .view(gloo.TextureFloat2D)
        self.program['u_observation'] = self.observation

    def compute_score(self, model_ids, model_trafos, extrinsics, intrinsics):
        # PREPARE RENDERING

        # compose model matrix:
        # - apply model space offset
        # - apply pose (in camera space)
        # - transform to world space
        # - transpose to be row major (OpenGL)
        mats_off = [self.models[model_id].mat_offset for model_id in model_ids]
        mats_model = [m.copy() for m in model_trafos]

        mat_world2cam = extrinsics.copy()
        R = mat_world2cam[:3, :3].T
        t = -R @ mat_world2cam[:3, 3]
        mat_cam2world = np.eye(4)
        mat_cam2world[:3, :3], mat_cam2world[:3, 3] = R, t

        mats_model = [(mat_cam2world @ mat_model @ mat_off).T for mat_model, mat_off in zip(mats_model, mats_off)]

        # prepare view and projection matrices (row major)
        mat_view = self._compute_view(mat_cam2world)  # gl view matrix from camera matrix
        mat_proj = self._compute_proj(intrinsics)  # projection matrix
        mat_view_proj = mat_view @ mat_proj  # view-projection matrix

        # STAGE 1) compute \hat{d}_T and \hat{n}_T: render model under estimated pose
        self.fbo_stage_1.activate()
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self.program['u_mode'] = 0

        for i, (model_id, m) in enumerate(zip(model_ids, mats_model)):
            self.program['u_mv'] = m @ mat_view
            self.program['u_mvp'] = m @ mat_view_proj

            try:
                model = self.models[model_id]
                model.draw(self.program)
            except ValueError as _:
                print("failed to draw")
                return 0

        # STAGE 2)
        # - compute sub-scores f_d(T) and f_n(T) per-pixel
        self.fbo_stage_2.activate()
        self.program['u_texture'] = self.normal_depth_buf
        self.program['u_mode'] = 1
        self.program['u_mv'] = np.eye(4)
        self.program['u_mvp'] = np.eye(4)
        self.models[-1].draw(self.program)

        # - compute verification score \bar{f}(T): speed-up summation by reading from mipmap layer
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.cost_buf.handle)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        layer = 4
        buf = np.zeros((int(self.height / (2 ** layer)), int(self.width / (2 ** layer)), 4), dtype=np.float32)
        per_pixel_fit = gl.glGetTexImage(gl.GL_TEXTURE_2D, layer, gl.GL_RGBA, gl.GL_FLOAT, buf)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        n_valid = per_pixel_fit[:, :, 0].sum()
        if n_valid == 0:  # no valid pixel -> minimal fit
            return 0
        else:
            mean_f_d = per_pixel_fit[:, :, 1].sum() / n_valid
            mean_f_n = per_pixel_fit[:, :, 2].sum() / n_valid
            return (mean_f_d + mean_f_n) * 0.5

    def _compute_view(self, cam):
        R, t = cam[:3, :3], cam[:3, 3]

        # camera coord axes
        z = R @ [0, 0, 1]
        z = -z / np.linalg.norm(z)
        y = R @ [0, -1, 0]
        y = y / np.linalg.norm(y)
        x = np.cross(y, z)
        x = x / np.linalg.norm(x)

        # invert to get view matrix
        view = np.eye(4)
        view[:3, :3] = np.vstack((x, y, z)).T
        view[3, :3] = -view[:3, :3].T @ t

        return view

    def _compute_proj(self, intrinsics):
        width, height = self.width, self.height
        near, far = self.near, self.far
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        s = intrinsics[0, 1]
        q = -(far + near) / (far - near)
        qn = -2 * far * near / (far - near)
        mat_proj = np.array([[2 * fx / width, -2 * s / width, (-2 * cx + width) / width, 0],
                             [0, 2 * fy / height, (2 * cy - height) / height, 0],
                             [0, 0, q, qn],
                             [0, 0, -1, 0]]).T
        return mat_proj
