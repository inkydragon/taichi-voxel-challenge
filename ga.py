from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(voxel_edges=0, exposure=1)
scene.set_directional_light((1, .3, .3), .8, (1, 1, 1))
scene.set_background_color((0, 0, 0))
scene.set_floor(-64, (0.01, 0.01, 0.012))

N = 64


@ti.func
def rgb(r, g, b):
    return vec3(r/255.0, g/255.0, b/255.0)


@ti.func
def gray(g):
    return rgb(g, g, g)


@ti.func
def draw_point(p, mat_body, col_body):
    mat_top = 2
    col_top = gray(255)
    for y in ti.ndrange(p.y):
        scene.set_voxel(vec3(p.x, y, p.z), mat_body, col_body)
    scene.set_voxel(p, mat_top, col_top)


@ti.func
def draw_line(z, mat, col):
    y = 5
    for x in ti.ndrange((-N, N)):
        y = (ti.random() * 10)
        draw_point(vec3(x, y, z), mat, col)


@ti.kernel
def initialize_voxels():
    mat = 1
    col = gray(100)
    for z in ti.ndrange((-N, N)):
        if (z % 3) == 0:
            draw_line(z, mat, col)

initialize_voxels()
scene.finish()
