# -*- coding: utf-8 -*-

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from contextlib import contextmanager
import os, sys
from typing import Optional, Type, Callable, Dict, List, Tuple

from abc import ABC
import numpy as np
import torch
import pycuda
from glumpy import app, gloo, gl, ext
import imgui
from kaolin.render.camera import Camera
from kaolin.io import utils
from kaolin.io import obj

from wisp.core.primitives import PrimitivesPack
from wisp.framework import WispState, watch
from wisp.renderer.core import RendererCore
from wisp.renderer.core.control import CameraControlMode, WispKey, WispMouseButton
from wisp.renderer.core.control import FirstPersonCameraMode, TrackballCameraMode, TurntableCameraMode
from wisp.renderer.gizmos import Gizmo, WorldGrid, AxisPainter, PrimitivesPainter
from wisp.renderer.gui import WidgetRendererProperties, WidgetGPUStats, WidgetSceneGraph, WidgetImgui
from wisp.ops.spc.conversions import mesh_to_spc

@contextmanager
def cuda_activate(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage.
        Boilerplate code based in part on pytorch-glumpy.
    """
    mapping = img.map()
    yield mapping.array(0, 0)
    mapping.unmap()

OPATH = os.path.normpath(os.path.join(__file__, "../../../../data/test/obj/1.obj"))

'''
kaolin-wisp\wisp\ops\spc\conversions.py
def mesh_to_spc(vertices, faces, level):
    """Construct SPC from a mesh.

    Args:
        vertices (torch.FloatTensor): Vertices of shape [V, 3]
        faces (torch.LongTensor): Face indices of shape [F, 3]
        level (int): The level of the octree

    Returns:
        (torch.ByteTensor, torch.ShortTensor, torch.LongTensor, torch.BoolTensor):
        - the octree tensor
        - point hierarchy
        - pyramid
        - prefix
    """
    octree = mesh_to_octree(vertices, faces, level)
    points, pyramid, prefix = octree_to_spc(octree)
    return octree, points, pyramid, prefix 
'''
def getObjLayers(f=OPATH, color = [[1, 0, 0, 1]], scale=10):
    mesh = obj.import_mesh(f,
             with_materials=True, with_normals=True,
             error_handler=obj.skip_error_handler,
             heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_naive_homogenize)

    vertices = mesh.vertices.cpu()
    faces = mesh.faces.cpu()
    level = 1

    if not len(vertices):
        return []
    # blows memory
    #octree, points, pyramid, prefix = mesh_to_spc(mesh.vertices, mesh.faces, level)
    points = vertices
    """ 
    uvs_list=[mesh.uvs.cpu()],
    face_uvs_idx_list=[mesh.face_uvs_idx.cpu()],
    uvs = mesh.uvs.cuda().unsqueeze(0)
    face_uvs_idx = mesh.face_uvs_idx.cuda()
    face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx).detach()
    face_uvs.requires_grad = False
    texture_map = torch.ones((1, 3, texture_res, texture_res), dtype=torch.float, device='cuda',
                            requires_grad=True)
    diffuse_color = mesh.materials[0]['map_Kd'] 
    """
    layers_to_draw = [PrimitivesPack()]
    start = torch.FloatTensor()
    end = torch.FloatTensor()
    colorT = torch.FloatTensor(color)   
    for i in range(0, len(faces)):
        face = faces[i]
        start = vertices[face[0]]
        end = vertices[face[1]]
        layers_to_draw[0].add_lines(start, end, colorT)
        start = vertices[face[1]]
        end = vertices[face[2]]
        layers_to_draw[0].add_lines(start, end, colorT)
        start = vertices[face[2]]
        end = vertices[face[0]]
        layers_to_draw[0].add_lines(start, end, colorT)

     # add points
    points_layers_to_draw = [PrimitivesPack()]
    color = [[0, 0, 1, 1]]
    colorT = torch.FloatTensor(color)   
    for i in range(0, len(points)):
        points_layers_to_draw[0].add_points(points[i], colorT)
   
    return layers_to_draw, points_layers_to_draw


class WispApp(ABC):
    dataset = None
    # Period of time between user interactions before resetting back to full resolution mode
    COOLDOWN_BETWEEN_RESOLUTION_CHANGES = 0.35  # In seconds

    def __init__(self, wisp_state, window_name="wisp app", dataset = None):
        # Initialize app state instance
        self.wisp_state: WispState = wisp_state
        self.init_wisp_state(wisp_state)

        # Create main app window & initialize GL context
        # glumpy with a specialized glfw backend takes care of that (backend is imgui integration aware)
        window = self._create_window(self.width, self.height, window_name)
        self.register_io_mappings()

        # Initialize gui, assumes the window is managed by glumpy with glfw
        imgui.create_context()
        self._is_imgui_focused = False
        self._is_imgui_hovered = False
        self._is_reposition_imgui_menu = True
        self.canvas_dirty = False
        self.dataset = dataset
        # Note: Normally pycuda.gl.autoinit should be invoked here after the window is created,
        # but wisp already initializes it when the library first loads. See wisp.__init__.py

        # Initialize applicative renderer, which independently paints images for the main canvas
        render_core = RendererCore(self.wisp_state)

        self.window = window                    # App window with a GL context & oversees event callbacks
        self.render_core = render_core          # Internal renderer, responsible for painting over canvas
        self.render_clock = app.clock.Clock()
        self.render_clock.tick()
        self.interactions_clock = app.clock.Clock()
        self.interactions_clock.tick()
        self._was_interacting_prev_frame = False

        # The initialization of these fields is deferred util "on_resize" is first prompted.
        # There we generate a simple billboard GL program with a shared CUDA resource
        # Canvas content will be blitted onto it
        self.cuda_buffer: Optional[pycuda.gl.RegisteredImage] = None    # CUDA buffer, as a shared resource with OpenGL
        self.depth_cuda_buffer: Optional[pycuda.gl.RegisteredImage] = None
        self.canvas_program: Optional[gloo.Program] = None              # GL program used to paint a single billboard

        self.user_mode: CameraControlMode = None

        self.widgets = self.create_widgets()        # Create gui widgets for this app
        self.gizmos = self.create_gizmos(data=None)          # Create canvas widgets for this app
        self.prim_painter = PrimitivesPainter() # grid
        # add a mesh, points
        layers, points_layers_to_draw = getObjLayers()
        # add points
        self.points = PrimitivesPainter()
        self.points.redraw(points_layers_to_draw)

        # draw mesh
        self.mesh = PrimitivesPainter()
        self.mesh.redraw(layers)

        self.register_event_handlers()
        self.change_user_mode(self.default_user_mode())

        self.redraw()

    def init_wisp_state(self, wisp_state: WispState) -> None:
        """ A hook for applications to initialize specific fields inside the wisp state object.
            This function is called before the entire renderer is constructed, hence initialized fields can
            be defined to affect the behaviour of the renderer.
        """
        # Channels available to view over the canvas
        wisp_state.renderer.available_canvas_channels = ["RGB"]
        wisp_state.renderer.selected_canvas_channel = "RGB"

    def create_widgets(self) -> List[WidgetImgui]:
        """ Override to define which widgets are used by the wisp app. """
        return [WidgetGPUStats(), WidgetRendererProperties(), WidgetSceneGraph()]

    def create_gizmos(self, data=None) -> Dict[str, Gizmo]:
        """ Override to define which gizmos are used by the wisp app. """
        gizmos = dict()
        planes = self.wisp_state.renderer.reference_grids
        axes = set(''.join(planes))
        grid_size = 10.0
        print(" ___create_gizmos ", data)
        for plane in planes:
            gizmos[f'world_grid_{plane}'] = WorldGrid(squares_per_axis=20, grid_size=grid_size,
                                                      line_color=(128, 128, 128), line_size=1, plane=plane)
            gizmos[f'world_grid_fine_{plane}'] = WorldGrid(squares_per_axis=200, grid_size=10.0,
                                                           line_color=(128, 128, 128), line_size=2, plane=plane)
        # Axes on top of the reference grid
        gizmos['grid_axis_painter'] = AxisPainter(axes_length=grid_size, line_width=1,
                                                  axes=axes, is_bidirectional=False)
        return gizmos

    def default_user_mode(self) -> str:
        """ Override to determine the default camera control mode """
        return "Turntable"

    def register_event_handlers(self) -> None:
        """ Override (and call super) to register additional event handlers """
        watch(watched_obj=self.wisp_state.renderer, field="cam_controller", status="changed",
              handler=self.on_cam_controller_changed)
        watch(watched_obj=self.wisp_state.renderer, field="selected_camera", status="changed",
              handler=self.on_selected_camera_changed)
        watch(watched_obj=self.wisp_state.renderer, field="selected_canvas_channel", status="changed",
              handler=self.on_selected_canvas_channel_changed)
        watch(watched_obj=self.wisp_state.renderer, field="selected_camera_lens", status="changed",
              handler=self.on_selected_camera_lens_changed)
        watch(watched_obj=self.wisp_state.renderer, field="clear_color_value", status="changed",
              handler=self.on_clear_color_value_changed)

    def on_cam_controller_changed(self, value: Type[CameraControlMode]):
        # Stay synced with state change: generate new instance of mode type
        mode_type = value
        self.user_mode = mode_type(render_core=self.render_core, wisp_state=self.wisp_state)

    def on_selected_camera_changed(self, value: Camera):
        # Rebuild camera controller to free any cached info from previous camera
        cam_controller_cls = type(self.user_mode)
        self.user_mode = cam_controller_cls(render_core=self.render_core, wisp_state=self.wisp_state)

        # Adjust the width / height according to current state of the renderer window
        self.render_core.resize_canvas(height=self.height, width=self.width)

    def on_selected_camera_lens_changed(self, value: str):
        self.render_core.change_camera_projection_mode(value)

    def on_selected_canvas_channel_changed(self, value: str):
        self.canvas_dirty = True    # Request canvas redraw

    def on_clear_color_value_changed(self, value: Tuple[float, float, float]):
        self.canvas_dirty = True    # Request canvas redraw

    def run(self):
        """ Initiate events message queue """
        app.run()   # App clock should always run as frequently as possible (background tasks should not be limited)

    def _create_window(self, width, height, window_name):
        # Currently assume glfw backend due to integration with imgui
        app.use("glfw_imgui")

        win_config = app.configuration.Configuration()
        if self.wisp_state.renderer.antialiasing == 'msaa_4x':
            win_config.samples = 4

        # glumpy implicitly sets the GL context as current
        window = app.Window(width=width, height=height, title=window_name, config=win_config)
        window.on_draw = self.on_draw
        window.on_resize = self.on_resize
        window.on_key_press = self.on_key_press
        window.on_key_release = self.on_key_release
        window.on_mouse_press = self.on_mouse_press
        window.on_mouse_drag = self.on_mouse_drag
        window.on_mouse_release = self.on_mouse_release
        window.on_mouse_scroll = self.on_mouse_scroll
        window.on_mouse_motion = self.on_mouse_motion

        if self.wisp_state.renderer.antialiasing == 'msaa_4x':
            gl.glEnable(gl.GL_MULTISAMPLE)

        return window

    @staticmethod
    def _create_gl_depth_billboard_program(texture: np.ndarray, depth_texture: np.ndarray):
        vertex = """
                    uniform float scale;
                    attribute vec2 position;
                    attribute vec2 texcoord;
                    varying vec2 v_texcoord;
                    void main()
                    {
                        v_texcoord = texcoord;
                        gl_Position = vec4(scale*position, 0.0, 1.0);
                    } """

        fragment = """
                    uniform sampler2D tex;
                    uniform sampler2D depth_tex;
                    varying vec2 v_texcoord;
                    void main()
                    {
                        gl_FragColor = texture2D(tex, v_texcoord);
                        gl_FragDepth = texture2D(depth_tex, v_texcoord).r;
                    } """
        # TODO (operel): r component is a waste?

        # Compile GL program
        canvas = gloo.Program(vertex, fragment, count=4)

        # Upload fixed values to GPU
        canvas['position'] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        canvas['texcoord'] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        canvas['scale'] = 1.0
        canvas['tex'] = texture
        canvas['depth_tex'] = depth_texture
        return canvas

    @staticmethod
    def _create_cugl_shared_texture(res_h, res_w, channel_depth, map_flags=pycuda.gl.graphics_map_flags.WRITE_DISCARD,
                                    dtype=np.uint8):
        """ Create and return a Texture2D with gloo and pycuda views. """
        if issubclass(dtype, np.integer):
            tex = np.zeros((res_h, res_w, channel_depth), dtype).view(gloo.Texture2D)
        elif issubclass(dtype, np.floating):
            tex = np.zeros((res_h, res_w, channel_depth), dtype).view(gloo.TextureFloat2D)
        else:
            raise ValueError(f'_create_cugl_shared_texture invoked with unsupported texture dtype: {dtype}')
        tex.activate()  # Force gloo to create on GPU
        tex.deactivate()
        cuda_buffer = pycuda.gl.RegisteredImage(int(tex.handle), tex.target,
                                                map_flags)  # Create shared GL / CUDA resource
        return tex, cuda_buffer

    def _reposition_gui_menu(self, menu_width, main_menu_height):
        window_height = self.window.height
        window_width = self.window.width
        imgui.set_next_window_size(width=menu_width, height=window_height-main_menu_height, condition=imgui.ALWAYS)
        imgui.set_next_window_position(x=window_width-menu_width, y=main_menu_height, condition=imgui.ALWAYS)
        self._is_reposition_imgui_menu = False

    def render_gui(self, state):
        imgui.new_frame()
        if imgui.begin_main_menu_bar():
            main_menu_height = imgui.get_window_height()
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", 'Cmd+Q', False, True
                )

                if clicked_quit:
                    exit(1)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        if self._is_reposition_imgui_menu:
            self._reposition_gui_menu(menu_width=350, main_menu_height=main_menu_height)
        imgui.begin("Scene Information", True)

        for widget in self.widgets:
            widget.paint(state)

        self._is_imgui_hovered = imgui.core.is_window_hovered(imgui.HOVERED_ANY_WINDOW |
                                                              imgui.HOVERED_ALLOW_WHEN_BLOCKED_BY_POPUP |
                                                              imgui.HOVERED_ALLOW_WHEN_BLOCKED_BY_ACTIVE_ITEM)
        self._is_imgui_hovered = self._is_imgui_hovered or \
                                 imgui.core.is_any_item_hovered() or imgui.is_any_item_active()
        imgui.end()

        self._is_imgui_focused = imgui.is_window_focused(imgui.FOCUS_ROOT_WINDOW)

        imgui.end_frame()
        imgui.render()

    def render_canvas(self, render_core, time_delta, force_render):
        renderbuffer = render_core.render(time_delta, force_render)
        buffer_attachment = renderbuffer.image().rgba
        buffer_attachment = buffer_attachment.flip([0])  # Flip y axis
        img = buffer_attachment.byte().contiguous()

        buffer_attachment_depth = renderbuffer.depth
        buffer_attachment_depth = buffer_attachment_depth.flip([0])
        depth_img = buffer_attachment_depth.repeat(1,1,4).contiguous()

        return img, depth_img

    def _blit_to_gl_renderbuffer(self, img, depth_img, canvas_program, cuda_buffer, depth_cuda_buffer, height):
        shared_tex = canvas_program['tex']
        shared_tex_depth = canvas_program['depth_tex']

        # copy from torch into buffer
        assert shared_tex.nbytes == img.numel() * img.element_size()
        assert shared_tex_depth.nbytes == depth_img.numel() * depth_img.element_size()    # TODO: using a 4d tex
        cpy = pycuda.driver.Memcpy2D()
        with cuda_activate(cuda_buffer) as ary:
            cpy.set_src_device(img.data_ptr())
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = shared_tex.nbytes // height
            cpy.height = height
            cpy(aligned=False)
        torch.cuda.synchronize()
        # TODO (operel): remove double synchronize after depth debug
        cpy = pycuda.driver.Memcpy2D()
        with cuda_activate(depth_cuda_buffer) as ary:
            cpy.set_src_device(depth_img.data_ptr())
            cpy.set_dst_array(ary)
            cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = shared_tex_depth.nbytes // height
            cpy.height = height
            cpy(aligned=False)
        torch.cuda.synchronize()

        canvas_program.draw(gl.GL_TRIANGLE_STRIP)

    def update_renderer_state(self, wisp_state, dt):
        """
        Populate the scene state object with the most recent information about the interactive renderer.
        The scene state, for example, may be used by the GUI widgets to display up to date information.
        This function is invoked in the beginning of the render() function, before the gui and the canvas are drawn.
        :param wisp_state The WispState object holding shared information between components about the wisp app.
        :param dt Amount of time elapsed since the last update.
        """
        wisp_state.renderer.fps = app.clock.get_fps()
        wisp_state.renderer.dt = dt
        wisp_state.renderer.cam_controller = type(self.user_mode)

    def change_user_mode(self, user_mode: str):
        if user_mode == 'Trackball':
            self.wisp_state.renderer.cam_controller = TrackballCameraMode
        elif user_mode == 'First Person View':
            self.wisp_state.renderer.cam_controller = FirstPersonCameraMode
        elif user_mode == 'Turntable':
            self.wisp_state.renderer.cam_controller = TurntableCameraMode

    @torch.no_grad()
    def redraw(self):
        # Let the renderer redraw the data layers if needed
        self.render_core.redraw()

        # Regenerate the GL primitives according to up-to-date data layers
        layers_to_draw = self.render_core.active_data_layers()
        self.prim_painter.redraw(layers_to_draw)

    @torch.no_grad()
    def render(self):
        dt = self.render_clock.tick()  # Tick render clock: dt is now the exact time elapsed since last render

        # Populate the scene state with the most recent information about the interactive renderer.
        # The scene state, for example, may be used by the GUI widgets to display up to date information.
        self.update_renderer_state(self.wisp_state, dt)

        # Clear color / depth buffers before rendering the next frame
        clear_color = (*self.wisp_state.renderer.clear_color_value, 1.0)    # RGBA
        self.window.clear(color=clear_color)

        # imgui renders first
        self.render_gui(self.wisp_state)

        if self.canvas_dirty:
            self.redraw()

        # Invoke the timer tick event, and let the camera controller update the state of any interactions
        # of the user which involve the time elapsed (i.e: velocity, acceleration of movements).
        self.user_mode.handle_timer_tick(dt)

        if self.user_mode.is_interacting() or self._is_imgui_hovered:
            self.render_core.set_low_resolution()
        else:
            # Allow a fraction of a second before turning full resolution on.
            # User interactions sometimes behave like a rapid burst of short and quick interactions.
            if self._was_interacting_prev_frame:
                self.interactions_clock.tick()
            time_since_last_interaction = self.interactions_clock.time() - self.interactions_clock.last_ts
            if time_since_last_interaction > self.COOLDOWN_BETWEEN_RESOLUTION_CHANGES:
                self.render_core.set_full_resolution()

        self._was_interacting_prev_frame = self.user_mode.is_interacting()

        # render canvas: core proceeds by invoking internal renderers tracers
        # output is rendered on a Renderbuffer object, backed by torch tensors
        img, depth_img = self.render_canvas(self.render_core, dt, self.canvas_dirty)

        # glumpy code injected within the pyimgui render loop to blit the rendercore output to the actual canvas
        # The torch buffers are copied by pycuda to CUDA buffers, connected as shared resources as 2d GL textures
        self._blit_to_gl_renderbuffer(img, depth_img, self.canvas_program, self.cuda_buffer,
                                      self.depth_cuda_buffer, self.height)

        # Finally, render OpenGL gizmos on the canvas.
        # This may include the world grid, or vectorial lines / points belonging to data layers
        camera = self.render_core.camera
        for gizmo in self.gizmos.values():
            gizmo.render(camera)
        self.prim_painter.render(camera)
        # mesh
        if (self.mesh.lines):
            self.mesh.render(camera)
        if (self.points.points):
            self.points.render(camera)
        self.canvas_dirty = False

    def register_background_task(self, hook: Callable[[], None]) -> None:
        def _run_hook(dt: float):
            if not self.wisp_state.renderer.background_tasks_paused:
                hook()
        self.window.on_idle = _run_hook

    def on_draw(self, dt=None):
        # dt arg comes from the app clock, the renderer clock is maintained separately from the background tasks
        # Interactive mode on, or interaction have just started
        is_interacting = self.wisp_state.renderer.interactive_mode or self.user_mode.is_interacting()
        if is_interacting or self.is_time_to_render():
            self.render()     # Render objects uploaded to GPU

    def is_time_to_render(self):
        time_since_last_render = self.render_clock.time() - self.render_clock.last_ts
        target_fps = self.wisp_state.renderer.target_fps
        if target_fps is None or ((target_fps > 0) and time_since_last_render >= (1 / target_fps)):
            return True
        return False

    def on_resize(self, width, height):
        self.width = width
        self.height = height

        # Handle pycuda shared resources
        if self.cuda_buffer is not None:
            del self.cuda_buffer    # TODO(operel): is this proper pycuda deallocation?
            del self.depth_cuda_buffer
        tex, cuda_buffer = self._create_cugl_shared_texture(height, width, self.channel_depth)
        depth_tex, depth_cuda_buffer = self._create_cugl_shared_texture(height, width, 4, dtype=np.float32)   # TODO: Single channel
        self.cuda_buffer = cuda_buffer
        self.depth_cuda_buffer = depth_cuda_buffer
        if self.canvas_program is None:
            self.canvas_program = self._create_gl_depth_billboard_program(texture=tex, depth_texture=depth_tex)
        else:
            if self.canvas_program['tex'] is not None:
                self.canvas_program['tex'].delete()
            if self.canvas_program['depth_tex'] is not None:
                self.canvas_program['depth_tex'].delete()
            self.canvas_program['tex'] = tex
            self.canvas_program['depth_tex'] = depth_tex

        self.render_core.resize_canvas(height=height, width=width)
        self.window.activate()
        gl.glViewport(0, 0, width, height)
        self._is_reposition_imgui_menu = True   # Signal menu it needs to shift after resize

    def is_canvas_event(self):
        """ Is canvas in focus or any of imgui's windows """
        return not self._is_imgui_focused

    def on_mouse_press(self, x, y, button):
        if self.is_canvas_event():
            self.user_mode.handle_mouse_press(x, y, button)

    def on_mouse_drag(self, x, y, dx, dy, button):
        if self.is_canvas_event():
            self.user_mode.handle_mouse_drag(x, y, dx, dy, button)

    def on_mouse_release(self, x, y, button):
        if self.is_canvas_event():
            self.user_mode.handle_mouse_release(x, y, button)

    def on_mouse_scroll(self, x, y, dx, dy):
        """ The mouse wheel was scrolled by (dx,dy). """
        if self.is_canvas_event():
            self.user_mode.handle_mouse_scroll(x, y, dx, dy)

    def on_mouse_motion(self, x, y, dx, dy):
        """ The mouse was moved with no buttons held down. """
        if self.is_canvas_event():
            self.user_mode.handle_mouse_motion(x, y, dx, dy)

    @property
    def width(self):
        return self.wisp_state.renderer.canvas_width

    @width.setter
    def width(self, value: int):
        self.wisp_state.renderer.canvas_width = value

    @property
    def height(self):
        return self.wisp_state.renderer.canvas_height

    @height.setter
    def height(self, value: int):
        self.wisp_state.renderer.canvas_height = value

    @property
    def channel_depth(self):
        return 4  # Assume the framebuffer keeps RGBA

    @property
    def canvas_dirty(self):
        return self.wisp_state.renderer.canvas_dirty

    @canvas_dirty.setter
    def canvas_dirty(self, value: bool):
        self.wisp_state.renderer.canvas_dirty = value

    def _update_imgui_keys(self, symbol):
        # Normally glfw shouldn't be explicitly imported as glumpy uses it as backend.
        # However, here we are forced to do that to take care of missing key mappings
        import glfw
        keys = [glfw.KEY_BACKSPACE, glfw.KEY_DELETE, glfw.KEY_ENTER, glfw.KEY_HOME, glfw.KEY_END,
                glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT,
                glfw.KEY_RIGHT, glfw.KEY_LEFT, glfw.KEY_UP, glfw.KEY_DOWN,
                glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4, glfw.KEY_5,
                glfw.KEY_6, glfw.KEY_7, glfw.KEY_8, glfw.KEY_9,
                glfw.KEY_KP_0, glfw.KEY_KP_1, glfw.KEY_KP_2, glfw.KEY_KP_3, glfw.KEY_KP_4, glfw.KEY_KP_5,
                glfw.KEY_KP_6, glfw.KEY_KP_7, glfw.KEY_KP_8, glfw.KEY_KP_9,
                glfw.KEY_KP_ENTER, glfw.KEY_KP_ADD, glfw.KEY_KP_SUBTRACT, glfw.KEY_KP_DECIMAL
                ]
        mappings = {    # Missing keys from glumpy glfw-imgui backend
            glfw.KEY_KP_0: glfw.KEY_0,
            glfw.KEY_KP_1: glfw.KEY_1,
            glfw.KEY_KP_2: glfw.KEY_2,
            glfw.KEY_KP_3: glfw.KEY_3,
            glfw.KEY_KP_4: glfw.KEY_4,
            glfw.KEY_KP_5: glfw.KEY_5,
            glfw.KEY_KP_6: glfw.KEY_6,
            glfw.KEY_KP_7: glfw.KEY_7,
            glfw.KEY_KP_8: glfw.KEY_8,
            glfw.KEY_KP_9: glfw.KEY_9,
            glfw.KEY_KP_ENTER: glfw.KEY_ENTER,
            glfw.KEY_KP_SUBTRACT: glfw.KEY_MINUS,
            glfw.KEY_KP_DECIMAL: glfw.KEY_PERIOD
        }
        updated_symbol = symbol
        for key in keys:
            is_key_on = glfw.get_key(self.window.native_window, key)
            imgui.get_io().keys_down[key] = is_key_on

            if symbol == -1 and is_key_on and key in mappings:
                updated_symbol = mappings[key]

        # TODO: Verify imgui keys have been properly mapped during initialization..
        # imgui.get_io().key_map[imgui.KEY_BACKSPACE] = app.window.key.BACKSPACE
        # imgui.get_io().key_map[imgui.KEY_DELETE] = app.window.key.DELETE
        # imgui.get_io().key_map[imgui.KEY_ENTER] = glfw.KEY_ENTER
        # imgui.get_io().key_map[imgui.KEY_RIGHT_ARROW] = glfw.KEY_RIGHT
        # imgui.get_io().key_map[imgui.KEY_LEFT_ARROW] = glfw.KEY_LEFT
        # imgui.get_io().key_map[imgui.KEY_UP_ARROW] = glfw.KEY_UP
        # imgui.get_io().key_map[imgui.KEY_DOWN_ARROW] = glfw.KEY_DOWN
        return updated_symbol

    def on_key_press(self, symbol, modifiers):
        symbol = self._update_imgui_keys(symbol)
        if symbol > 0:
            imgui.get_io().add_input_character(symbol)

        if self.is_canvas_event():
            self.user_mode.handle_key_press(symbol, modifiers)

            # TODO: Shouldn't be here
            cam_mode = None
            if symbol in (app.window.key.T, ord('T'), ord('t')):
                cam_mode = "Trackball"
            elif symbol in (app.window.key.F, ord('F'), ord('f')):
                cam_mode = "First Person View"
            elif symbol in (app.window.key.U, ord('U'), ord('u')):
                cam_mode = "Turntable"
            if cam_mode is not None:
                self.change_user_mode(cam_mode)

    def on_key_release(self, symbol, modifiers):
        symbol = self._update_imgui_keys(symbol)

        if self.is_canvas_event():
            self.user_mode.handle_key_release(symbol, modifiers)

    def dump_framebuffer(self, path='./framebuffer'):
        framebuffer = np.zeros((self.width, self.height * 3), dtype=np.uint8)
        gl.glReadPixels(0, 0, self.width, self.height,
                        gl.GL_RGB, gl.GL_UNSIGNED_BYTE, framebuffer)
        framebuffer = np.flip(framebuffer, 0)
        ext.png.from_array(framebuffer, 'RGB').save(path + '_color.png')

        framebuffer = np.zeros((self.width, self.height), dtype=np.float32)
        gl.glReadPixels(0, 0, self.width, self.height,
                        gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, framebuffer)
        framebuffer = np.flip(framebuffer, 0)
        ext.png.from_array(framebuffer, 'L').save(path + '_depth.png')


    def register_io_mappings(self):
        WispMouseButton.register_symbol(WispMouseButton.LEFT_BUTTON, app.window.mouse.LEFT)
        WispMouseButton.register_symbol(WispMouseButton.MIDDLE_BUTTON, app.window.mouse.MIDDLE)
        WispMouseButton.register_symbol(WispMouseButton.RIGHT_BUTTON, app.window.mouse.RIGHT)

        WispKey.register_symbol(WispKey.LEFT, app.window.key.LEFT)
        WispKey.register_symbol(WispKey.RIGHT, app.window.key.RIGHT)
        WispKey.register_symbol(WispKey.UP, app.window.key.UP)
        WispKey.register_symbol(WispKey.DOWN, app.window.key.DOWN)

        # TODO: Take care of remaining mappings, and verify the event handlers of glumpy were not overriden