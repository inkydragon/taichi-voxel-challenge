from scene import Scene
import taichi as ti
from taichi.math import *

# ==== Constant Def ====
# --- Grid Const
MIN_GRID_IDX = 0
MAX_GRID_IDX = 64
HALF_GRID = MAX_GRID_IDX / 2

# --- Material Constant
CLEAR_MAT = 0
SOLID_MAT = 1
LIGHT_MAT = 2

# --- Color Constant
WHITE = ivec3(255) / 255
GRAY = ivec3(128) / 255
RED = ivec3(255, 0, 0) / 255


# ---- 场景设置
scene = Scene(exposure=1)
scene.set_directional_light((1, 1, 1), 0.1, (1, 1, 1))
scene.set_background_color((0.3, 0.4, 0.6))


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


@ti.func
def fill(p, mat=SOLID_MAT, col=GRAY):
    assert is_in_grid(p)
    scene.set_voxel(p, mat, col)


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
    fill_shell(ivec3(0), ivec3(63), mat, col)
    slice_box(ivec3(0, 0, 63), ivec3(63))
    # fill_box(ivec3(HALF_GRID, 0, HALF_GRID), ivec3(HALF_GRID, 64, HALF_GRID), LIGHT_MAT, RED)
    for y in ti.ndrange(32):
        scene.set_voxel(ivec3(32, y, 32), LIGHT_MAT, RED)


if __name__ == '__main__':
    initialize_voxels()
    scene.finish()
