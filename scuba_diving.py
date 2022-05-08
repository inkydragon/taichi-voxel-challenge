from scene import Scene
import taichi as ti
from taichi.math import *

# --- Material Constant
CLEAR_MAT = 0
SOLID_MAT = 1
LIGHT_MAT = 2


# ---- 场景设置
scene = Scene(voxel_edges=0.06, exposure=1)
# scene.set_directional_light((1, 1, 1), 0.1, (1, 1, 1))
# scene.set_background_color((0.3, 0.4, 0.6))
scene.set_floor(-64, (1, 1, 1))


# Color
@ti.func
def rgb(r, g, b):
    return vec3(r/255.0, g/255.0, b/255.0)


@ti.func  # Input [0, 128) grid => set vox in [-64, 64)
def fill(p, mat=SOLID_MAT, col=vec3(0.8)):
    # scene.set_voxel(p - HALF_GRID, mat, col)
    scene.set_voxel(p, mat, col)


@ti.func  # remove vox
def remove(p):
    fill(p, mat=CLEAR_MAT)


@ti.func  # z 轴颜色渐变填充
def z_grad_fill_box(x, y, mat):
    col0 = rgb(3, 58, 80)
    col1 = rgb(57, 166, 202)
    col_rate = 0.2
    col2 = vec3(1)
    lo, up = ti.min(x, y), ti.max(x, y)+1
    _steps = up.z - lo.z
    # up.z <--- Z ---- lo.z :  col0 (a=1) -- col1 @ col_rate --- col2 (a=0)
    for i, j, k in ti.ndrange((lo.x, up.x), (lo.y, up.y), (lo.z, up.z)):
        a = (k - lo.z) / _steps
        if a <= col_rate:  # a in [0, col_rate]
            alpha = 1.0 - (a/col_rate)
            fill(vec3(i, j, k), mat, mix(col1, col2, alpha))
        else:  # a in [col_rate, 1.0]
            alpha = 1.0 - ((a-col_rate)/(1-col_rate))
            fill(vec3(i, j, k), mat, mix(col0, col1, alpha))


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
    z_grad_fill_box(ivec3(-62), ivec3(62), LIGHT_MAT)
    slice_x_cone(c0=vec3(0, 0, 63), r0=40, c1=vec3(0, 0, -63), r1=10)


initialize_voxels()
scene.finish()
