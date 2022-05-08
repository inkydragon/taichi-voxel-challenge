from scene import Scene
import taichi as ti
from taichi.math import *

# ==== Constant Def ====
# --- Grid Const
MIN_GRID_IDX = 0
MAX_GRID_IDX = 128
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
scene.set_floor(-64, (1, 1, 1))


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
    # assert is_in_grid(p)
    # scene.set_voxel(p - HALF_GRID, mat, col)
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

@ti.func  # 圆形去除. x: 圆心, r: 半径
def slice_xy_circle(c, r):
    x_lo, x_hi = ti.floor(c.x - r), ti.floor(c.x + r)
    y_lo, y_hi = ti.floor(c.y - r), ti.floor(c.y + r)
    z0 = c.z
    for x, y in ti.ndrange((x_lo, x_hi+1), (y_lo, y_hi+1)):
        if (x*x+y*y) <= r*r:
            remove(vec3(x, y, z0))

@ti.func  # z 轴圆台（圆锥）去除. (c0, r0) 圆台底面, (c1, r1) 圆台顶部
def slice_x_cone(c0, r0, c1, r1):
    assert c0.z >= c1.z
    _steps = c0.z - c1.z + 1
    for _step_i in ti.ndrange((0, _steps)):
        a = _step_i*1.0 / _steps
        c = mix(c1, c0, a)
        r = mix(r1, r0, a)
        slice_xy_circle(c, r)

# ==== Main loop
@ti.kernel
def initialize_voxels():
    mat = SOLID_MAT
    col = gray(200)
    fill_shell(ivec3(0), ivec3(127), mat, col)
    slice_box(ivec3(0, 0, 127), ivec3(127))
    # fill_box(ivec3(HALF_GRID, 0, HALF_GRID), ivec3(HALF_GRID, 64, HALF_GRID), LIGHT_MAT, RED)
    for y in ti.ndrange(64):
        scene.set_voxel(ivec3(0, -y, 0), LIGHT_MAT, RED)
    slice_xy_circle(vec3(0,0,63), 60)
    slice_x_cone(vec3(0,0,63), 60, vec3(0,0,-60), 2)



if __name__ == '__main__':
    initialize_voxels()
    scene.finish()
