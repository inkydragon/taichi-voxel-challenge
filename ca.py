from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=1)
scene.set_directional_light((1, 1, 1), 0.1, (1, 1, 1))
scene.set_background_color((0.3, 0.4, 0.6))
scene.set_floor(-64, (1, 1, 1))

# ---- Const
MIN_GRID_IDX = 0
MAX_GRID_IDX = 128
HALF_GRID = MAX_GRID_IDX / 2
SOLID_MAT = 1
CLEAR_MAT = 0

GRAY = ivec3(128)


# Color
@ti.func
def rgb(r, g, b):
    return vec3(r/255.0, g/255.0, b/255.0)


@ti.func
def gray(g):
    return rgb(g, g, g)


# ---- Vox helper func
@ti.func  # is in grad [0, 128)^3
def is_in_grid(p):
    return any(MIN_GRID_IDX <= p < MAX_GRID_IDX)


@ti.func  # Input [0, 128) grid => set vox in [-64, 64)
def fill(p, mat=SOLID_MAT, col=GRAY):
    assert is_in_grid(p)
    scene.set_voxel(p - HALF_GRID, mat, col)


@ti.func  # remove vox
def remove(p):
    fill(p, mat=CLEAR_MAT)


@ti.func  # 填充两点确定的长方体
def fill_box(x, y, mat, col):
    lo, up = ti.min(x, y), ti.max(x, y)+1
    for i, j, k in ti.ndrange((lo.x, up.x), (lo.y, up.y), (lo.z, up.z)):
        fill(vec3(i, j, k), mat, col)


@ti.func  # 填充两点确定的长方体
def slice_box(x, y):
    fill_box(x, y, mat=CLEAR_MAT, col=GRAY)


@ti.func  # 填充两点确定的长方体外壳，内部留空
def fill_shell(x, y, mat, col):
    fill_box(x, y, mat, col)
    slice_box(x+1, y-1)


# ==== Main loop
@ti.kernel
def initialize_voxels():
    mat = SOLID_MAT
    col = gray(200)
    fill_shell(ivec3(0), ivec3(127), mat, col)
    slice_box(ivec3(0, 0, 127), ivec3(127))
    for y in ti.ndrange(64):
        scene.set_voxel(ivec3(0, -y, 0), mat, col)


initialize_voxels()
scene.finish()
