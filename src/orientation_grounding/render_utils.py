# Originally from https://github.com/SpatialVision/Orient-Anything

import typing as t
from functools import partial

import numpy as np
from copy import deepcopy

from PIL import Image, ImageColor, ImageFilter
from math import sqrt


def speedup_normalize(x, y, z):
    unit = sqrt(x * x + y * y + z * z)
    if unit == 0:
        return 0, 0, 0
    return x / unit, y / unit, z / unit


def get_min_max(a, b, c):
    min = a
    max = a
    if min > b:
        min = b
    if min > c:
        min = c
    if max < b:
        max = b
    if max < c:
        max = c
    return int(min), int(max)

def speedup_dot_product(a0, a1, a2, b0, b1, b2):
    r = a0 * b0 + a1 * b1 + a2 * b2
    return r


def speedup_cross_product(a0, a1, a2, b0, b1, b2):
    x = a1 * b2 - a2 * b1
    y = a2 * b0 - a0 * b2
    z = a0 * b1 - a1 * b0
    return x,y,z


# @cython.boundscheck(False)
def speedup_generate_faces(triangles, width, height):
    """ draw the triangle faces with z buffer

    Args:
        triangles: groups of vertices

    FYI:
        * zbuffer, https://github.com/ssloy/tinyrenderer/wiki/Lesson-3:-Hidden-faces-removal-(z-buffer)
        * uv mapping and perspective correction
    """
    i, j, k, length     = 0, 0, 0, 0
    bcy, bcz, x, y, z   = 0.,0.,0.,0.,0.
    a, b, c             = [0.,0.,0.],[0.,0.,0.],[0.,0.,0.]
    m, bc               = [0.,0.,0.],[0.,0.,0.]
    uva, uvb, uvc       = [0.,0.],[0.,0.],[0.,0.]
    minx, maxx, miny, maxy = 0,0,0,0
    length = triangles.shape[0]
    zbuffer = {}
    faces = []

    for i in range(length):
        a = triangles[i, 0, 0], triangles[i, 0, 1], triangles[i, 0, 2]
        b = triangles[i, 1, 0], triangles[i, 1, 1], triangles[i, 1, 2]
        c = triangles[i, 2, 0], triangles[i, 2, 1], triangles[i, 2, 2]
        uva = triangles[i, 0, 3], triangles[i, 0, 4]
        uvb = triangles[i, 1, 3], triangles[i, 1, 4]
        uvc = triangles[i, 2, 3], triangles[i, 2, 4]
        minx, maxx = get_min_max(a[0], b[0], c[0])
        miny, maxy = get_min_max(a[1], b[1], c[1])
        pixels = []
        for j in range(minx, maxx + 2):
            for k in range(miny - 1, maxy + 2):
                # 必须显式转换成 double 参与底下的运算，不然结果是错的
                x = j
                y = k

                m[0], m[1], m[2] = speedup_cross_product(c[0] - a[0], b[0] - a[0], a[0] - x, c[1] - a[1], b[1] - a[1], a[1] - y)
                if abs(m[2]) > 0:
                    bcy = m[1] / m[2]
                    bcz = m[0] / m[2]
                    bc = (1 - bcy - bcz, bcy, bcz)
                else:
                    continue

                # here, -0.00001 because of the precision lose
                if bc[0] < -0.00001 or bc[1] < -0.00001 or bc[2] < -0.00001:
                    continue

                z = 1 / (bc[0] / a[2] + bc[1] / b[2] + bc[2] / c[2])

                # Blender 导出来的 uv 数据，跟之前的顶点数据有一样的问题，Y轴是个反的，
                # 所以这里的纹理图片要旋转一下才能 work
                v = (uva[0] * bc[0] / a[2] + uvb[0] * bc[1] / b[2] + uvc[0] * bc[2] / c[2]) * z * width
                u = height - (uva[1] * bc[0] / a[2] + uvb[1] * bc[1] / b[2] + uvc[1] * bc[2] / c[2]) * z * height

                # https://en.wikipedia.org/wiki/Pairing_function
                idx = ((x + y) * (x + y + 1) + y) / 2
                if zbuffer.get(idx) is None or zbuffer[idx] < z:
                    zbuffer[idx] = z
                    pixels.append((i, j, k, int(u) - 1, int(v) - 1))

        faces.append(pixels)
    return faces



class Canvas:
    def __init__(self, filename=None, height=500, width=500):
        self.filename = filename
        self.height, self.width = height, width
        self.img = Image.new("RGBA", (self.height, self.width), (0, 0, 0, 0))

    def draw(self, dots, color: t.Union[tuple, str]):
        if isinstance(color, str):
            color = ImageColor.getrgb(color)
        if isinstance(dots, tuple):
            dots = [dots]
        for dot in dots:
            if dot[0]>=self.height or dot[1]>=self.width or dot[0]<0 or dot[1]<0:
                # print(dot)
                continue
            self.img.putpixel(dot, color + (255,))

    def add_white_border(self, border_size=5):
        # 确保输入图像是 RGBA 模式
        if self.img.mode != "RGBA":
            self.img = self.img.convert("RGBA")
        
        # 提取 alpha 通道
        alpha = self.img.getchannel("A")
        # print(alpha.size)
        dilated_alpha = alpha.filter(ImageFilter.MaxFilter(size=border_size * 2 + 1))
        # # print(dilated_alpha.size)
        white_area = Image.new("RGBA", self.img.size, (255, 255, 255, 255))
        white_area.putalpha(dilated_alpha)
        
        # 合并膨胀后的白色区域与原图像
        result = Image.alpha_composite(white_area, self.img)
        # expanded_alpha = ImageOps.expand(alpha, border=border_size, fill=255)
        # white_border = Image.new("RGBA", self.img.size, (255, 255, 255, 255))
        # white_border.putalpha(alpha)
        return result

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # self.img = add_white_border(self.img)
        self.img.save(self.filename)
        pass

# 2D part


class Vec2d:
    __slots__ = "x", "y", "arr"

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], Vec3d):
            self.arr = Vec3d.narr
        else:
            assert len(args) == 2
            self.arr = list(args)

        self.x, self.y = [d if isinstance(d, int) else int(d + 0.5) for d in self.arr]

    def __repr__(self):
        return f"Vec2d({self.x}, {self.y})"

    def __truediv__(self, other):
        return (self.y - other.y) / (self.x - other.x)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


def draw_line(
    v1: Vec2d, v2: Vec2d, canvas: Canvas, color: t.Union[tuple, str] = "white"
):
    """
    Draw a line with a specified color

    https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    """
    v1, v2 = deepcopy(v1), deepcopy(v2)
    if v1 == v2:
        canvas.draw((v1.x, v1.y), color=color)
        return

    steep = abs(v1.y - v2.y) > abs(v1.x - v2.x)
    if steep:
        v1.x, v1.y = v1.y, v1.x
        v2.x, v2.y = v2.y, v2.x
    v1, v2 = (v1, v2) if v1.x < v2.x else (v2, v1)
    slope = abs((v1.y - v2.y) / (v1.x - v2.x))
    y = v1.y
    error: float = 0
    incr = 1 if v1.y < v2.y else -1
    dots = []
    for x in range(int(v1.x), int(v2.x + 0.5)):
        dots.append((int(y), x) if steep else (x, int(y)))
        error += slope
        if abs(error) >= 0.5:
            y += incr
            error -= 1

    canvas.draw(dots, color=color)


def draw_triangle(v1, v2, v3, canvas, color, wireframe=False):
    """
    Draw a triangle with 3 ordered vertices

    http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
    """
    _draw_line = partial(draw_line, canvas=canvas, color=color)

    if wireframe:
        _draw_line(v1, v2)
        _draw_line(v2, v3)
        _draw_line(v1, v3)
        return

    def sort_vertices_asc_by_y(vertices):
        return sorted(vertices, key=lambda v: v.y)

    def fill_bottom_flat_triangle(v1, v2, v3):
        invslope1 = (v2.x - v1.x) / (v2.y - v1.y)
        invslope2 = (v3.x - v1.x) / (v3.y - v1.y)

        x1 = x2 = v1.x
        y = v1.y

        while y <= v2.y:
            _draw_line(Vec2d(x1, y), Vec2d(x2, y))
            x1 += invslope1
            x2 += invslope2
            y += 1

    def fill_top_flat_triangle(v1, v2, v3):
        invslope1 = (v3.x - v1.x) / (v3.y - v1.y)
        invslope2 = (v3.x - v2.x) / (v3.y - v2.y)

        x1 = x2 = v3.x
        y = v3.y

        while y > v2.y:
            _draw_line(Vec2d(x1, y), Vec2d(x2, y))
            x1 -= invslope1
            x2 -= invslope2
            y -= 1

    v1, v2, v3 = sort_vertices_asc_by_y((v1, v2, v3))

    # 填充
    if v1.y == v2.y == v3.y:
        pass
    elif v2.y == v3.y:
        fill_bottom_flat_triangle(v1, v2, v3)
    elif v1.y == v2.y:
        fill_top_flat_triangle(v1, v2, v3)
    else:
        v4 = Vec2d(int(v1.x + (v2.y - v1.y) / (v3.y - v1.y) * (v3.x - v1.x)), v2.y)
        fill_bottom_flat_triangle(v1, v2, v4)
        fill_top_flat_triangle(v2, v4, v3)


# 3D part


class Vec3d:
    __slots__ = "x", "y", "z", "arr"

    def __init__(self, *args):
        # for Vec4d cast
        if len(args) == 1 and isinstance(args[0], Vec4d):
            vec4 = args[0]
            arr_value = (vec4.x, vec4.y, vec4.z)
        else:
            assert len(args) == 3
            arr_value = args
        self.arr = np.array(arr_value, dtype=np.float64)
        self.x, self.y, self.z = self.arr

    def __repr__(self):
        return repr(f"Vec3d({','.join([repr(d) for d in self.arr])})")

    def __sub__(self, other):
        return self.__class__(*[ds - do for ds, do in zip(self.arr, other.arr)])

    def __bool__(self):
        """ False for zero vector (0, 0, 0)
        """
        return any(self.arr)


class Mat4d:
    def __init__(self, narr=None, value=None):
        self.value = np.matrix(narr) if value is None else value

    def __repr__(self):
        return repr(self.value)

    def __mul__(self, other):
        return self.__class__(value=self.value * other.value)


class Vec4d(Mat4d):
    def __init__(self, *narr, value=None):
        if value is not None:
            self.value = value
        elif len(narr) == 1 and isinstance(narr[0], Mat4d):
            self.value = narr[0].value
        else:
            assert len(narr) == 4
            self.value = np.matrix([[d] for d in narr])

        self.x, self.y, self.z, self.w = (
            self.value[0, 0],
            self.value[1, 0],
            self.value[2, 0],
            self.value[3, 0],
        )
        self.arr = self.value.reshape((1, 4))

class Model:
    def __init__(self, filename, texture_filename):
        """
        https://en.wikipedia.org/wiki/Wavefront_.obj_file#Vertex_normal_indices
        """
        self.vertices = []
        self.uv_vertices = []
        self.uv_indices = []
        self.indices = []

        texture = Image.open(texture_filename)
        self.texture_array = np.array(texture)
        self.texture_width, self.texture_height = texture.size

        with open(filename) as f:
            for line in f:
                if line.startswith("v "):
                    x, y, z = [float(d) for d in line.strip("v").strip().split(" ")]
                    self.vertices.append(Vec4d(x, y, z, 1))
                elif line.startswith("vt "):
                    u, v = [float(d) for d in line.strip("vt").strip().split(" ")]
                    self.uv_vertices.append([u, v])
                elif line.startswith("f "):
                    facet = [d.split("/") for d in line.strip("f").strip().split(" ")]
                    self.indices.append([int(d[0]) for d in facet])
                    self.uv_indices.append([int(d[1]) for d in facet])


# Math util
def normalize(v: Vec3d):
    return Vec3d(*speedup_normalize(*v.arr))


def dot_product(a: Vec3d, b: Vec3d):
    return speedup_dot_product(*a.arr, *b.arr)


def cross_product(a: Vec3d, b: Vec3d):
    return Vec3d(*speedup_cross_product(*a.arr, *b.arr))

BASE_LIGHT = 0.9
def get_light_intensity(face) -> float:
    # lights = [Vec3d(-2, 4, -10), Vec3d(10, 4, -2), Vec3d(8, 8, -8), Vec3d(0, 0, -8)]
    lights = [Vec3d(-2, 4, -10)]
    # lights = []

    v1, v2, v3 = face
    up = normalize(cross_product(v2 - v1, v3 - v1))
    intensity = BASE_LIGHT
    for light in lights:
        intensity += dot_product(up, normalize(light))*0.2
    return intensity


def look_at(eye: Vec3d, target: Vec3d, up: Vec3d = Vec3d(0, -1, 0)) -> Mat4d:
    """
    http://www.songho.ca/opengl/gl_camera.html#lookat

    Args:
        eye: 摄像机的世界坐标位置
        target: 观察点的位置
        up: 就是你想让摄像机立在哪个方向
            https://stackoverflow.com/questions/10635947/what-exactly-is-the-up-vector-in-opengls-lookat-function
            这里默认使用了 0, -1, 0， 因为 blender 导出来的模型数据似乎有问题，导致y轴总是反的，于是把摄像机的up也翻一下得了。
    """
    f = normalize(eye - target)
    l = normalize(cross_product(up, f))  # noqa: E741
    u = cross_product(f, l)

    rotate_matrix = Mat4d(
        [[l.x, l.y, l.z, 0], [u.x, u.y, u.z, 0], [f.x, f.y, f.z, 0], [0, 0, 0, 1.0]]
    )
    translate_matrix = Mat4d(
        [[1, 0, 0, -eye.x], [0, 1, 0, -eye.y], [0, 0, 1, -eye.z], [0, 0, 0, 1.0]]
    )

    return Mat4d(value=(rotate_matrix * translate_matrix).value)


def perspective_project(r, t, n, f, b=None, l=None):  # noqa: E741
    """
    目的：
        把相机坐标转换成投影在视网膜的范围在(-1, 1)的笛卡尔坐标

    原理：
        对于x，y坐标，相似三角形可以算出投影点的x，y
        对于z坐标，是假设了near是-1，far是1，然后带进去算的
        http://www.songho.ca/opengl/gl_projectionmatrix.html
        https://www.scratchapixel.com/lessons/3d-basic-rendering/perspective-and-orthographic-projection-matrix/opengl-perspective-projection-matrix

    推导出来的矩阵：
        [
            2n/(r-l) 0        (r+l/r-l)   0
            0        2n/(t-b) (t+b)/(t-b) 0
            0        0        -(f+n)/f-n  (-2*f*n)/(f-n)
            0        0        -1          0
        ]

    实际上由于我们用的视网膜(near pane)是个关于远点对称的矩形，所以矩阵简化为：
        [
            n/r      0        0           0
            0        n/t      0           0
            0        0        -(f+n)/f-n  (-2*f*n)/(f-n)
            0        0        -1          0
        ]

    Args:
        r: right, t: top, n: near, f: far, b: bottom, l: left
    """
    return Mat4d(
        [
            [n / r, 0, 0, 0],
            [0, n / t, 0, 0],
            [0, 0, -(f + n) / (f - n), (-2 * f * n) / (f - n)],
            [0, 0, -1, 0],
        ]
    )


def draw(screen_vertices, world_vertices, model, canvas, wireframe=True):
    """standard algorithm
    """
    for triangle_indices in model.indices:
        vertex_group = [screen_vertices[idx - 1] for idx in triangle_indices]
        face = [Vec3d(world_vertices[idx - 1]) for idx in triangle_indices]
        if wireframe:
            draw_triangle(*vertex_group, canvas=canvas, color="black", wireframe=True)
        else:
            intensity = get_light_intensity(face)
            if intensity > 0:
                draw_triangle(
                    *vertex_group, canvas=canvas, color=(int(intensity * 255),) * 3
                )


def draw_with_z_buffer(screen_vertices, world_vertices, model, canvas):
    """ z-buffer algorithm
    """
    intensities = []
    triangles = []
    for i, triangle_indices in enumerate(model.indices):
        screen_triangle = [screen_vertices[idx - 1] for idx in triangle_indices]
        uv_triangle = [model.uv_vertices[idx - 1] for idx in model.uv_indices[i]]
        world_triangle = [Vec3d(world_vertices[idx - 1]) for idx in triangle_indices]
        intensities.append(abs(get_light_intensity(world_triangle)))
        # take off the class to let Cython work
        triangles.append(
            [np.append(screen_triangle[i].arr, uv_triangle[i]) for i in range(3)]
        )

    faces = speedup_generate_faces(
        np.array(triangles, dtype=np.float64), model.texture_width, model.texture_height
    )
    for face_dots in faces:
        for dot in face_dots:
            intensity = intensities[dot[0]]
            u, v = dot[3], dot[4]
            color = model.texture_array[u, v]
            canvas.draw((dot[1], dot[2]), tuple(int(c * intensity) for c in color[:3]))
            # TODO: add object rendering mode (no texture)
            # canvas.draw((dot[1], dot[2]), (int(255 * intensity),) * 3)


def render(model, height, width, filename, cam_loc, wireframe=False):
    """
    Args:
        model: the Model object
        height: cavas height
        width: cavas width
        picname: picture file name
    """
    model_matrix = Mat4d([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # TODO: camera configration
    view_matrix = look_at(Vec3d(cam_loc[0], cam_loc[1], cam_loc[2]), Vec3d(0, 0, 0))
    projection_matrix = perspective_project(0.5, 0.5, 3, 1000)

    world_vertices = []

    def mvp(v):
        world_vertex = model_matrix * v
        world_vertices.append(Vec4d(world_vertex))
        return projection_matrix * view_matrix * world_vertex

    def ndc(v):
        """
        各个坐标同时除以 w，得到 NDC 坐标
        """
        v = v.value
        w = v[3, 0]
        x, y, z = v[0, 0] / w, v[1, 0] / w, v[2, 0] / w
        return Mat4d([[x], [y], [z], [1 / w]])

    def viewport(v):
        x = y = 0
        w, h = width, height
        n, f = 0.3, 1000
        return Vec3d(
            w * 0.5 * v.value[0, 0] + x + w * 0.5,
            h * 0.5 * v.value[1, 0] + y + h * 0.5,
            0.5 * (f - n) * v.value[2, 0] + 0.5 * (f + n),
        )

    # the render pipeline
    screen_vertices = [viewport(ndc(mvp(v))) for v in model.vertices]

    with Canvas(filename, height, width) as canvas:
        if wireframe:
            draw(screen_vertices, world_vertices, model, canvas)
        else:
            draw_with_z_buffer(screen_vertices, world_vertices, model, canvas)
        
        render_img = canvas.add_white_border().copy()
    return render_img