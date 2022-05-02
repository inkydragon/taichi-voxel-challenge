from scene import Scene
import taichi as ti
from taichi.math import *

scene = Scene(exposure=1)
scene.set_directional_light((1, 1, 1), 0.1, (1, 1, 1))
scene.set_background_color((0.3, 0.4, 0.6))

@ti.func
def rgb(r, g, b):
    return vec3(r/255.0, g/255.0, b/255.0)
@ti.func
def gray(g):
    return rgb(g, g, g)

@ti.func
def randAxisUnitVec():
    rnd06 = ti.floor(6 * ti.random())
    unitVec = ivec3(0, 0, 0)
    if rnd06 == 0:
        unitVec = ivec3(-1, 0, 0)
    elif rnd06 == 1:
        unitVec = ivec3( 1, 0, 0)
    elif rnd06 == 2:
        unitVec = ivec3( 0,-1, 0)
    elif rnd06 == 3:
        unitVec = ivec3( 0, 1, 0)
    elif rnd06 == 4:
        unitVec = ivec3( 0, 0,-1)
    elif rnd06 == 5:
        unitVec = ivec3( 0, 0, 1)
    return unitVec

@ti.func
def pointInBox(p):
    return any(ivec3(0,0,0) <= p <= ivec3(60,60,120))

# 填充两点确定的长方体
@ti.func
def fillBox(x, y, mat, rgb):
    l = ti.min(x,y)
    u = ti.max(x,y)
    for i, j, k in ti.ndrange((l.x, u.x), (l.y, u.y), (l.z, u.z)):
        scene.set_voxel(vec3(i, j, k), mat, rgb)

@ti.func
def worm(startVec, size, moveRepeat, mat, rgb):
    pos = startVec
    moves = 0
    while pointInBox(pos):
        minCorner = ti.floor(pos - 0.5*size)
        maxCorner = minCorner + size
        fillBox(minCorner, maxCorner, mat, rgb)
        move = ivec3(0,0,0)
        if (0==(moves % moveRepeat)):
            move = randAxisUnitVec() * size
        pos += move
        moves += 1

@ti.kernel
def initialize_voxels():
    n = 60
    mat = 1
    rgb = gray(200)
    # for i in ti.ndrange(2):
    worm(ivec3(30,30,0),
        # size=[1, 4]
        1+ti.floor(ti.random()),
        # moveRepeat=[2, 4]
        2 + ti.floor(2*ti.random()),
        mat, rgb
    )
    # worm(ivec3(32, 32, 0), 1, 2 + ti.floor(2 * ti.random()), mat, rgb)


initialize_voxels()
scene.finish()
