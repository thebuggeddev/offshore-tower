"""Procedural offshore observation tower.

Run with Blender 4.5 LTS:
  blender -b --python Offshore_Tower_Generator.py -- --output-dir ./tower_output --render

Every structural assembly is generated from geometry. Procedural node materials are
baked to glTF-compatible base-color, ORM and normal maps before the GLB is exported.
No downloaded models, image planes, or external texture assets are used.
"""
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

PI = math.pi
TAU = 2.0 * PI
RNG = random.Random(24051986)
ARGS = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'tower_output'))
parser.add_argument('--render', action='store_true')
parser.add_argument('--texture-size', type=int, default=512)
parser.add_argument('--samples', type=int, default=32)
opts = parser.parse_args(ARGS)
OUT = Path(opts.output_dir).resolve()
TEX = OUT / 'textures'
VIEWS = OUT / 'verification'
for folder in (OUT, TEX, VIEWS):
    folder.mkdir(parents=True, exist_ok=True)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
for datablocks in (bpy.data.materials, bpy.data.curves, bpy.data.meshes, bpy.data.cameras, bpy.data.lights):
    for block in list(datablocks):
        if block.users == 0:
            datablocks.remove(block)

SCENE = bpy.context.scene
SCENE.unit_settings.system = 'METRIC'
SCENE.unit_settings.scale_length = 1.0
COLLECTIONS = {}
for name in (
    '01_Foundation', '02_Lower_Deck', '03_Utility_Rooms', '04_Tapered_Shaft',
    '05_Service_Platforms', '06_Control_Room', '07_Roof', '08_Communications',
    '09_Crane', '10_Mechanical_Details', '11_Markings', '12_Internal_Structure',
    '90_Presentation'
):
    coll = bpy.data.collections.new(name)
    SCENE.collection.children.link(coll)
    COLLECTIONS[name] = coll
CURRENT = COLLECTIONS['01_Foundation']


def use_collection(name):
    global CURRENT
    CURRENT = COLLECTIONS[name]


def link_object(obj):
    CURRENT.objects.link(obj)
    return obj


def srgb(hex_value):
    h = hex_value.lstrip('#')
    vals = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in vals)


def simple_material(name, color, roughness=0.6, metallic=0.0, transmission=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    rgba = (*srgb(color), 1.0)
    bsdf.inputs['Base Color'].default_value = rgba
    bsdf.inputs['Roughness'].default_value = roughness
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Transmission Weight'].default_value = transmission
    mat.diffuse_color = rgba
    return mat


def bake_emission_image(nodes, links, source_socket, image, plane):
    emission = nodes.get('Bake_Emission')
    links.new(source_socket, emission.inputs['Color'])
    target = nodes.new('ShaderNodeTexImage')
    target.image = image
    target.location = (700, -500)
    nodes.active = target
    for node in nodes:
        node.select = False
    target.select = True
    bpy.context.view_layer.objects.active = plane
    bpy.ops.object.select_all(action='DESELECT')
    plane.select_set(True)
    bpy.ops.object.bake(type='EMIT')
    nodes.remove(target)
    return image


def save_image(image, path, noncolor=False):
    # Color space is assigned at image creation. Changing it on a generated image
    # after baking can clear its pixel buffer in some Blender versions.
    image.filepath_raw = str(path)
    image.file_format = 'PNG'
    image.save()


def baked_pbr(name, palette, metallic, rough_range, streakiness=1.0, rust=0.25):
    """Bake procedural color/roughness/height nodes; derive a tangent normal map."""
    size = opts.texture_size
    # A reusable UV plane is the baking target, never part of the delivered tower.
    mesh = bpy.data.meshes.new(name + '_BakeMesh')
    mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])
    mesh.uv_layers.new(name='UVMap')
    for loop, uv in zip(mesh.uv_layers.active.data, [(0, 0), (1, 0), (1, 1), (0, 1)]):
        loop.uv = uv
    plane = bpy.data.objects.new(name + '_BakePlane', mesh)
    COLLECTIONS['90_Presentation'].objects.link(plane)
    plane.location = (0, 0, -80)

    proc = bpy.data.materials.new(name + '_Procedural_Source')
    proc.use_nodes = True
    nodes = proc.node_tree.nodes
    links = proc.node_tree.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    emission.name = 'Bake_Emission'
    links.new(emission.outputs[0], out.inputs['Surface'])
    uv = nodes.new('ShaderNodeTexCoord')
    broad_map = nodes.new('ShaderNodeVectorMath')
    broad_map.operation = 'MULTIPLY'
    broad_map.inputs[1].default_value = (1.2, 1.2, 1.0)
    links.new(uv.outputs['UV'], broad_map.inputs[0])
    broad = nodes.new('ShaderNodeTexNoise')
    broad.inputs['Scale'].default_value = 4.2
    broad.inputs['Detail'].default_value = 5.0
    broad.inputs['Roughness'].default_value = 0.72
    links.new(broad_map.outputs[0], broad.inputs['Vector'])
    streak_map = nodes.new('ShaderNodeVectorMath')
    streak_map.operation = 'MULTIPLY'
    streak_map.inputs[1].default_value = (8.0 * streakiness, 0.38, 1.0)
    links.new(uv.outputs['UV'], streak_map.inputs[0])
    streak = nodes.new('ShaderNodeTexNoise')
    streak.inputs['Scale'].default_value = 3.4
    streak.inputs['Detail'].default_value = 3.5
    streak.inputs['Roughness'].default_value = 0.68
    links.new(streak_map.outputs[0], streak.inputs['Vector'])
    mix = nodes.new('ShaderNodeMixRGB')
    mix.blend_type = 'MULTIPLY'
    mix.inputs[0].default_value = min(0.92, 0.38 + rust)
    links.new(broad.outputs['Fac'], mix.inputs[1])
    links.new(streak.outputs['Fac'], mix.inputs[2])
    ramp = nodes.new('ShaderNodeValToRGB')
    colors = [srgb(c) for c in palette]
    cr = ramp.color_ramp
    cr.elements.remove(cr.elements[1])
    cr.elements[0].position = 0.08
    cr.elements[0].color = (*colors[0], 1)
    for pos, color in zip((0.22, 0.36, 0.62), colors[1:]):
        el = cr.elements.new(pos)
        el.color = (*color, 1)
    links.new(mix.outputs[0], ramp.inputs[0])
    fine = nodes.new('ShaderNodeTexNoise')
    fine.inputs['Scale'].default_value = 175.0
    fine.inputs['Detail'].default_value = 2.4
    fine.inputs['Roughness'].default_value = 0.75
    links.new(uv.outputs['UV'], fine.inputs['Vector'])
    rough = nodes.new('ShaderNodeMapRange')
    rough.inputs['From Min'].default_value = 0.12
    rough.inputs['From Max'].default_value = 0.85
    rough.inputs['To Min'].default_value = rough_range[0]
    rough.inputs['To Max'].default_value = rough_range[1]
    links.new(broad.outputs['Fac'], rough.inputs['Value'])
    orm = nodes.new('ShaderNodeCombineXYZ')
    orm.inputs['X'].default_value = 1.0
    orm.inputs['Z'].default_value = metallic
    links.new(rough.outputs['Result'], orm.inputs['Y'])
    height = nodes.new('ShaderNodeMixRGB')
    height.blend_type = 'MULTIPLY'
    height.inputs[0].default_value = 0.32
    links.new(fine.outputs['Fac'], height.inputs[1])
    links.new(streak.outputs['Fac'], height.inputs[2])

    plane.data.materials.append(proc)
    SCENE.render.engine = 'CYCLES'
    SCENE.cycles.samples = 1
    SCENE.render.bake.use_clear = True
    SCENE.render.bake.margin = 4
    base = bpy.data.images.new(name + '_BaseColor', width=size, height=size, alpha=False)
    bake_emission_image(nodes, links, ramp.outputs['Color'], base, plane)
    save_image(base, TEX / (name + '_BaseColor.png'))
    packed_orm = bpy.data.images.new(name + '_ORM', width=size, height=size, alpha=False)
    packed_orm.colorspace_settings.name = 'Non-Color'
    bake_emission_image(nodes, links, orm.outputs[0], packed_orm, plane)
    save_image(packed_orm, TEX / (name + '_ORM.png'), noncolor=True)
    height_img = bpy.data.images.new(name + '_Height', width=size, height=size, alpha=False)
    height_img.colorspace_settings.name = 'Non-Color'
    bake_emission_image(nodes, links, height.outputs[0], height_img, plane)
    save_image(height_img, TEX / (name + '_Height.png'), noncolor=True)

    # Derive a seamless tangent-space normal map from the baked height field.
    pixels = np.empty(size * size * 4, dtype=np.float32)
    height_img.pixels.foreach_get(pixels)
    h = pixels.reshape(size, size, 4)[:, :, 0]
    dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 3.2
    dy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 3.2
    normal = np.stack((-dx, -dy, np.ones_like(h)), axis=-1)
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-6)
    normal = normal * 0.5 + 0.5
    rgba = np.ones((size, size, 4), dtype=np.float32)
    rgba[:, :, :3] = normal
    normal_img = bpy.data.images.new(name + '_Normal', width=size, height=size, alpha=False)
    normal_img.colorspace_settings.name = 'Non-Color'
    normal_img.pixels.foreach_set(rgba.astype(np.float32).ravel())
    save_image(normal_img, TEX / (name + '_Normal.png'), noncolor=True)

    # Rebuild with only image-based Principled inputs for predictable glTF export.
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    ns = mat.node_tree.nodes
    lk = mat.node_tree.links
    bsdf = ns.get('Principled BSDF')
    base_node = ns.new('ShaderNodeTexImage')
    base_node.image = base
    base_node.location = (-650, 250)
    lk.new(base_node.outputs['Color'], bsdf.inputs['Base Color'])
    orm_node = ns.new('ShaderNodeTexImage')
    orm_node.image = packed_orm
    orm_node.location = (-650, -30)
    split = ns.new('ShaderNodeSeparateColor')
    split.mode = 'RGB'
    split.location = (-380, -30)
    lk.new(orm_node.outputs['Color'], split.inputs[0])
    lk.new(split.outputs['Green'], bsdf.inputs['Roughness'])
    lk.new(split.outputs['Blue'], bsdf.inputs['Metallic'])
    norm_node = ns.new('ShaderNodeTexImage')
    norm_node.image = normal_img
    norm_node.location = (-650, -340)
    norm = ns.new('ShaderNodeNormalMap')
    norm.inputs['Strength'].default_value = 0.7
    norm.location = (-350, -300)
    lk.new(norm_node.outputs['Color'], norm.inputs['Color'])
    lk.new(norm.outputs[0], bsdf.inputs['Normal'])
    for img in (base, packed_orm, normal_img):
        img.pack()
    mat.diffuse_color = (*colors[2], 1)
    bpy.data.objects.remove(plane, do_unlink=True)
    bpy.data.materials.remove(proc)
    return mat


print('BAKING_PBR', flush=True)
ORANGE = baked_pbr('Orange_Steel', ['48251A', 'B54814', 'EB6D1C', 'F58C3B'], 0.28, (0.5, 0.84), 1.3, 0.32)
DARK = baked_pbr('Dark_Structural_Steel', ['171817', '302C26', '42423C', '686357'], 0.78, (0.46, 0.83), 0.8, 0.48)
ROOF = baked_pbr('Weathered_Roof', ['35342F', '70401F', '7E8177', 'B08A62'], 0.7, (0.48, 0.88), 1.0, 0.68)
RUST = baked_pbr('Oxidized_Joints', ['211B16', '4A2919', '79411F', 'A4612E'], 0.36, (0.72, 0.96), 1.0, 0.75)
CONCRETE = baked_pbr('Saltworn_Foundation', ['34362F', '55594E', '7B7C6B', 'A3A08A'], 0.0, (0.84, 0.98), 0.25, 0.15)
GLASS = simple_material('Smoked_Industrial_Glass', '243639', 0.2, 0.28, 0.1)
GLASS_2 = simple_material('Faded_Glass_Reflection', '425356', 0.26, 0.22, 0.08)
WHITE = simple_material('Aged_Ceramic_White', 'DDDCD0', 0.58, 0.08)
LOGO_WHITE = simple_material('Marking_Offwhite', 'F0E8D6', 0.66, 0.05)
RUBBER = simple_material('Cable_Rubber', '161918', 0.88, 0.0)
BLUE = simple_material('Faded_Blue_Equipment', '49636A', 0.72, 0.35)
RED = simple_material('Beacon_Red', '8F2F20', 0.28, 0.2)
SIGN_DARK = simple_material('Faded_Sign_Ink', '29312C', 0.8, 0.0)

# Shared mesh templates keep repeated bolts, rails and rungs linked in the GLB.
MESH_CACHE = {}


def cube_mesh(unique=False, dims=(1, 1, 1)):
    # Boolean-cut and beveled boxes must own their mesh; repeated unit boxes link.
    key = ('cube', 'unit')
    if not unique and key in MESH_CACHE:
        return MESH_CACHE[key]
    dx, dy, dz = (d / 2 for d in (dims if unique else (1, 1, 1)))
    verts = [(-dx,-dy,-dz),(dx,-dy,-dz),(dx,dy,-dz),(-dx,dy,-dz),
             (-dx,-dy,dz),(dx,-dy,dz),(dx,dy,dz),(-dx,dy,dz)]
    faces = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
    mesh = bpy.data.meshes.new('Linked_Box' if not unique else 'Beveled_Box')
    mesh.from_pydata(verts, [], faces)
    mesh.uv_layers.new(name='UVMap')
    face_uv = [(0,0),(1,0),(1,1),(0,1)]
    for poly in mesh.polygons:
        for li, uv in zip(poly.loop_indices, face_uv):
            mesh.uv_layers.active.data[li].uv = uv
    if not unique:
        MESH_CACHE[key] = mesh
    return mesh


def cylinder_mesh(segments=16):
    key = ('cylinder', segments)
    if key in MESH_CACHE:
        return MESH_CACHE[key]
    verts = []
    for z in (-0.5, 0.5):
        for i in range(segments):
            a = TAU * i / segments
            verts.append((math.cos(a), math.sin(a), z))
    faces = [tuple(reversed(range(segments))), tuple(range(segments, 2 * segments))]
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((i, j, j + segments, i + segments))
    mesh = bpy.data.meshes.new('Linked_Cylinder_' + str(segments))
    mesh.from_pydata(verts, [], faces)
    mesh.uv_layers.new(name='UVMap')
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index % segments
            z = mesh.vertices[mesh.loops[li].vertex_index].co.z + 0.5
            mesh.uv_layers.active.data[li].uv = (vi / segments, z)
    for poly in mesh.polygons[2:]:
        poly.use_smooth = True
    MESH_CACHE[key] = mesh
    return mesh


def sphere_mesh(segments=24, rings=12):
    key = ('sphere', segments, rings)
    if key in MESH_CACHE:
        return MESH_CACHE[key]
    verts = [(0, 0, 1)]
    for j in range(1, rings):
        p = PI * j / rings
        for i in range(segments):
            a = TAU * i / segments
            verts.append((math.sin(p)*math.cos(a), math.sin(p)*math.sin(a), math.cos(p)))
    verts.append((0, 0, -1))
    bottom = len(verts) - 1
    faces = []
    for i in range(segments):
        faces.append((0, 1 + i, 1 + (i + 1) % segments))
    for j in range(rings - 2):
        start = 1 + j * segments
        nxt = start + segments
        for i in range(segments):
            ni = (i + 1) % segments
            faces.append((start+i, nxt+i, nxt+ni, start+ni))
    start = 1 + (rings - 2) * segments
    for i in range(segments):
        faces.append((start+i, bottom, start+(i+1)%segments))
    mesh = bpy.data.meshes.new('Linked_Sphere')
    mesh.from_pydata(verts, [], faces)
    mesh.uv_layers.new(name='UVMap')
    for poly in mesh.polygons:
        poly.use_smooth = True
        for li in poly.loop_indices:
            co = mesh.vertices[mesh.loops[li].vertex_index].co
            mesh.uv_layers.active.data[li].uv = ((math.atan2(co.y, co.x) / TAU) % 1, math.acos(max(-1,min(1,co.z))) / PI)
    MESH_CACHE[key] = mesh
    return mesh


def object_from_mesh(name, mesh, location, material, scale=(1,1,1), rotation=None):
    obj = link_object(bpy.data.objects.new(name, mesh))
    obj.location = location
    obj.scale = scale
    if rotation is not None:
        if len(rotation) == 4:
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = rotation
        else:
            obj.rotation_euler = rotation
    if material:
        if len(obj.data.materials) == 0:
            obj.data.materials.append(material)
        obj.material_slots[0].link = 'OBJECT'
        obj.material_slots[0].material = material
    return obj


def box(name, location, dims, material=DARK, rotation=0.0, bevel=0.0, unique=False):
    obj = object_from_mesh(name, cube_mesh(unique, dims), location, material,
                           (1,1,1) if unique else dims, (0,0,rotation))
    if bevel:
        mod = obj.modifiers.new('Soft fabricated edges', 'BEVEL')
        mod.width = bevel
        mod.segments = 2
        mod = obj.modifiers.new('Weighted corner normals', 'WEIGHTED_NORMAL')
    return obj


def cylinder(name, location, radius, depth, material=DARK, segments=16, rotation=None):
    return object_from_mesh(name, cylinder_mesh(segments), location, material,
                            (radius, radius, depth), rotation)


def rod(name, start, end, radius=0.03, material=DARK, segments=12):
    a, b = Vector(start), Vector(end)
    delta = b-a
    return cylinder(name, (a+b)/2, radius, delta.length, material, segments,
                    delta.to_track_quat('Z','Y'))


def beam_between(name, start, end, width=0.08, depth=None, material=DARK):
    a, b = Vector(start), Vector(end)
    delta = b-a
    obj = box(name, (a+b)/2, (width, depth or width, delta.length), material)
    obj.rotation_mode = 'QUATERNION'
    obj.rotation_quaternion = delta.to_track_quat('Z','Y')
    return obj


def sphere(name, location, radius, material=DARK, scale=(1,1,1)):
    return object_from_mesh(name, sphere_mesh(), location, material,
                            tuple(radius*s for s in scale))


def mesh_object(name, verts, faces, material, uvs=None, smooth=False):
    mesh = bpy.data.meshes.new(name + '_Mesh')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.uv_layers.new(name='UVMap')
    for poly in mesh.polygons:
        poly.use_smooth = smooth
        for li in poly.loop_indices:
            vi = mesh.loops[li].vertex_index
            if uvs:
                mesh.uv_layers.active.data[li].uv = uvs[vi]
            else:
                co = mesh.vertices[vi].co
                mesh.uv_layers.active.data[li].uv = (co.x * 0.35, co.z * 0.35)
    return object_from_mesh(name, mesh, (0,0,0), material)


def curve_path(name, points, radius=0.015, material=RUBBER, cyclic=False):
    data = bpy.data.curves.new(name + '_Curve', 'CURVE')
    data.dimensions = '3D'
    data.resolution_u = 2
    data.bevel_depth = radius
    data.bevel_resolution = 2 if radius > 0.015 else 1
    spline = data.splines.new('POLY')
    spline.points.add(len(points)-1)
    for p, co in zip(spline.points, points):
        p.co = (*co, 1)
    spline.use_cyclic_u = cyclic
    obj = link_object(bpy.data.objects.new(name, data))
    data.materials.append(material)
    return obj


def sag_wire(name, start, end, sag=0.3, radius=0.01, material=RUBBER, samples=24):
    a, b = Vector(start), Vector(end)
    points = []
    for i in range(samples+1):
        t = i/samples
        p = a.lerp(b, t)
        p.z -= sag * 4*t*(1-t)
        points.append(p)
    return curve_path(name, points, radius, material)


def oct_ring(name, inner, outer, z, thickness, material=DARK, segments=8, offset=PI/8):
    verts, uvs = [], []
    for radius, zz in ((outer,z),(inner,z),(outer,z-thickness),(inner,z-thickness)):
        for i in range(segments):
            a = offset + TAU*i/segments
            verts.append((radius*math.cos(a),radius*math.sin(a),zz))
            uvs.append((i/segments*4, (radius-inner)*0.4))
    faces = []
    for i in range(segments):
        j = (i+1)%segments
        faces.extend([(i,j,segments+j,segments+i),
                      (2*segments+j,2*segments+i,3*segments+i,3*segments+j),
                      (j,i,2*segments+i,2*segments+j),
                      (segments+i,segments+j,3*segments+j,3*segments+i)])
    return mesh_object(name, verts, faces, material, uvs)


def face_frame(angle, radius, z=0):
    n = Vector((math.cos(angle), math.sin(angle), 0))
    t = Vector((-math.sin(angle), math.cos(angle), 0))
    apothem = radius * math.cos(PI/8)
    center = n*apothem + Vector((0,0,z))
    half = radius * math.sin(PI/8)
    return center, n, t, half


def boolean_cut(obj, cutter):
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new('Physical opening', 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver = 'EXACT'
    mod.object = cutter
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def add_window(name, center, n, t, width, height, mullions=1):
    phi = math.atan2(t.y, t.x)
    outer = center + n*0.105
    glass = box(name+'_RecessedGlass', center-n*0.035, (width-0.08,0.025,height-0.08),
                GLASS if RNG.random() < 0.75 else GLASS_2, phi)
    for s in (-1,1):
        box(name+'_FrameVertical', outer+t*(s*(width/2-0.025)), (0.055,0.07,height+0.08), DARK, phi)
        box(name+'_FrameHorizontal', outer+Vector((0,0,s*(height/2-0.025))), (width+0.06,0.07,0.055), DARK, phi)
    for i in range(1,mullions+1):
        x = -width/2 + width*i/(mullions+1)
        box(name+'_Mullion', outer+t*x, (0.035,0.06,height-0.05), DARK, phi)
    if height > 0.7:
        box(name+'_Transom', outer+Vector((0,0,height*0.12)), (width-0.04,0.06,0.035), DARK, phi)
    box(name+'_Sill', outer+Vector((0,0,-height/2-0.055))+n*0.035,
        (width+0.15,0.16,0.055), DARK, phi)
    return glass


def railing_segment(name, start, end, z, height=0.78, spacing=0.7, toe=True):
    a, b = Vector((start[0],start[1],z)), Vector((end[0],end[1],z))
    length = (b-a).length
    count = max(1, math.ceil(length/spacing))
    for i in range(count+1):
        p = a.lerp(b,i/count)
        rod(name+'_Post', p, p+Vector((0,0,height)), 0.025, DARK, 10)
        cylinder(name+'_Foot', p+Vector((0,0,0.035)), 0.055, 0.035, RUST, 12)
    for h,r in ((height,0.028),(height*0.52,0.018),(0.15,0.016)):
        rod(name+'_Rail', a+Vector((0,0,h)), b+Vector((0,0,h)), r, DARK, 10)
    if toe:
        beam_between(name+'_ToeBoard', a+Vector((0,0,0.08)), b+Vector((0,0,0.08)), 0.06,0.12,DARK)


def railing_polygon(name, radius, z, height=0.78, segments=8, offset=PI/8):
    pts = [(radius*math.cos(offset+TAU*i/segments),radius*math.sin(offset+TAU*i/segments)) for i in range(segments)]
    for i in range(segments):
        railing_segment(name, pts[i], pts[(i+1)%segments], z, height)


def add_bolt(name, position, normal, radius=0.024):
    p, n = Vector(position), Vector(normal).normalized()
    cylinder(name+'_Washer', p, radius*1.5, 0.012, RUST, 12, n.to_track_quat('Z','Y'))
    cylinder(name+'_Hex', p+n*0.014, radius, 0.026, DARK, 6, n.to_track_quat('Z','Y'))


# -----------------------------------------------------------------------------
# A. Offshore foundation: six load paths, complete bracing on all four sides.
# -----------------------------------------------------------------------------
print('BUILDING_FOUNDATION', flush=True)
use_collection('01_Foundation')
# Irregular, low seabed/concrete island with a modeled edge.
verts = [(0,0,0.09)]
segments = 56
for i in range(segments):
    a = TAU*i/segments
    r = 3.25 + RNG.uniform(-0.22,0.22)
    verts.append((r*math.cos(a),r*math.sin(a)*0.8,RNG.uniform(-0.01,0.06)))
faces = [(0,1+i,1+(i+1)%segments) for i in range(segments)]
island = mesh_object('Weathered_Seabed_Island', verts, faces, CONCRETE)
solid = island.modifiers.new('Island depth', 'SOLIDIFY')
solid.thickness = 0.14

leg_tops = [(-1.85,-1.55,4.25),(1.85,-1.55,4.25),(1.85,1.55,4.25),(-1.85,1.55,4.25)]
leg_bottoms = [(-2.45,-1.95,0.26),(2.45,-1.95,0.26),(2.45,1.95,0.26),(-2.45,1.95,0.26)]
for i,(bottom,top) in enumerate(zip(leg_bottoms,leg_tops)):
    box('Concrete_Pile_Cap', (bottom[0],bottom[1],0.19), (0.86,0.82,0.28), CONCRETE, bevel=0.05, unique=True)
    rod('Main_Battered_Leg_%02d'%i, bottom, top, 0.22, DARK, 24)
    for z in (0.48,3.82):
        p = Vector(bottom).lerp(Vector(top),(z-bottom[2])/(top[2]-bottom[2]))
        cylinder('Leg_Bolted_Collar', p, 0.265, 0.11, RUST, 24)
        for j in range(8):
            a = TAU*j/8
            add_bolt('Foundation_Anchor', p+Vector((0.26*math.cos(a),0.26*math.sin(a),0)), (math.cos(a),math.sin(a),0),0.021)
    box('Top_Gusset_Plate', (top[0],top[1],4.05), (0.62,0.54,0.12), RUST, bevel=0.02, unique=True)
# Two central vertical columns visible through the brace grid.
for x in (-0.62,0.62):
    rod('Central_Vertical_Column', (x,0.25,0.24), (x,0.25,4.28), 0.19, DARK, 24)
    box('Central_Column_Foot', (x,0.25,0.19), (0.7,0.7,0.24), CONCRETE, unique=True)

# Horizontal load-distribution frame.
for z in (1.0,3.92):
    corners = [(-2.15,-1.78,z),(2.15,-1.78,z),(2.15,1.78,z),(-2.15,1.78,z)]
    for i in range(4):
        beam_between('Perimeter_Horizontal_Girder', corners[i], corners[(i+1)%4], 0.16,0.2,DARK)
# X-braces on front, rear and both sides; flat steel rather than wire rods.
brace_sides = [
    ((-2.25,-1.8,0.85),(2.0,-1.68,3.83),(2.25,-1.8,0.85),(-2.0,-1.68,3.83)),
    ((-2.25,1.8,0.85),(2.0,1.68,3.83),(2.25,1.8,0.85),(-2.0,1.68,3.83)),
    ((-2.3,-1.72,0.85),(-1.95,1.65,3.83),(-2.3,1.72,0.85),(-1.95,-1.65,3.83)),
    ((2.3,-1.72,0.85),(1.95,1.65,3.83),(2.3,1.72,0.85),(1.95,-1.65,3.83)),
]
for side in brace_sides:
    beam_between('Heavy_Diagonal_Cross_Brace',side[0],side[1],0.11,0.16,DARK)
    beam_between('Heavy_Diagonal_Cross_Brace',side[2],side[3],0.11,0.16,DARK)
# Internal plan bracing, visible from below.
beam_between('Underdeck_Diagonal',(-1.9,-1.55,3.96),(1.9,1.55,3.96),0.1,0.14,DARK)
beam_between('Underdeck_Diagonal',(-1.9,1.55,3.96),(1.9,-1.55,3.96),0.1,0.14,DARK)

# -----------------------------------------------------------------------------
# B. Lower maintenance deck, plate/grating, perimeter rails and underdeck beams.
# -----------------------------------------------------------------------------
use_collection('02_Lower_Deck')
box('Lower_Deck_Structural_Plate',(0,0,4.28),(6.25,4.85,0.22),DARK,unique=True)
for y in (-2.18,2.18):
    box('Deck_Longitudinal_IBeam',(0,y,4.02),(6.05,0.16,0.34),DARK)
for x in (-2.72,-1.35,0,1.35,2.72):
    box('Deck_Cross_Beam',(x,0,4.03),(0.14,4.55,0.3),DARK)
# Actual raised grating bars around the exposed deck margins.
for i in range(35):
    x = -3.0 + i*0.176
    for y in (-2.12,2.12):
        box('Open_Grating_Bearing_Bar',(x,y,4.415),(0.024,0.48,0.055),DARK)
for y in (-2.33,-1.92,1.92,2.33):
    box('Grating_Cross_Tie',(0,y,4.43),(6.0,0.018,0.025),RUST)
# Complete rectangular guardrail with a front access gap.
rail_z = 4.42
railing_segment('LowerDeck_FrontLeft',(-3.04,-2.34),(-0.58,-2.34),rail_z)
railing_segment('LowerDeck_FrontRight',(0.58,-2.34),(3.04,-2.34),rail_z)
railing_segment('LowerDeck_Rear',(-3.04,2.34),(3.04,2.34),rail_z)
railing_segment('LowerDeck_Left',(-3.04,-2.34),(-3.04,2.34),rail_z)
railing_segment('LowerDeck_Right',(3.04,-2.34),(3.04,2.34),rail_z)
for x in (-2.8,2.8):
    for y in (-2.0,2.0):
        beam_between('Deck_Underside_Knee',(x,y,3.55),(x*0.72,y*0.72,4.18),0.09,0.12,RUST)
# Foundation access ladder with stand-off brackets.
for x in (-0.27,0.27):
    rod('Foundation_Access_Ladder_Rail',(x,-2.02,0.42),(x,-2.02,4.48),0.028,DARK)
for i in range(15):
    z = 0.55+i*0.27
    rod('Foundation_Access_Ladder_Rung',(-0.28,-2.02,z),(0.28,-2.02,z),0.02,DARK)
for z in (1.0,2.2,3.4):
    for x in (-0.27,0.27):
        rod('Ladder_Standoff',(x,-2.02,z),(x,-1.72,z),0.022,DARK)

# -----------------------------------------------------------------------------
# C. Lower orange utility rooms with real window/door openings and equipment.
# -----------------------------------------------------------------------------
use_collection('03_Utility_Rooms')

def utility_wall(name, location, dims, rotation, openings):
    wall = box(name,location,dims,ORANGE,rotation,unique=True)
    n = Vector((-math.sin(rotation+PI/2),-math.cos(rotation+PI/2),0))
    # local +x is the wall tangent; local -y is its outward direction.
    tangent = Vector((math.cos(rotation),math.sin(rotation),0))
    outward = Vector((math.sin(rotation),-math.cos(rotation),0))
    for idx,(x,z,w,h,kind) in enumerate(openings):
        c = Vector(location)+tangent*x+Vector((0,0,z-location[2]))
        cutter = box('Temporary_Opening',c,(w,0.65,h),None,rotation,unique=True)
        boolean_cut(wall,cutter)
        if kind == 'window':
            add_window(name+'_Window',c,outward,tangent,w,h,2 if w>0.8 else 1)
        else:
            # Recessed metal door, hinges, handle and threshold.
            box(name+'_Door',c-outward*0.035,(w-0.07,0.06,h-0.04),BLUE,rotation)
            for sx in (-1,1):
                box(name+'_DoorJamb',c+outward*0.05+tangent*(sx*w/2),(0.06,0.09,h+0.09),DARK,rotation)
            box(name+'_DoorHeader',c+outward*0.05+Vector((0,0,h/2)),(w+0.1,0.09,0.07),DARK,rotation)
            for zz in (-0.55,0.55):
                cylinder(name+'_DoorHinge',c+tangent*(-w*0.43)+outward*0.07+Vector((0,0,zz)),0.025,0.12,RUST,12)
            rod(name+'_DoorHandle',c+tangent*(w*0.3)+outward*0.1+Vector((0,0,-0.1)),
                c+tangent*(w*0.3)+outward*0.1+Vector((0,0,0.12)),0.018,WHITE)
            box(name+'_Threshold',c+outward*0.12+Vector((0,0,-h/2)),(w+0.12,0.3,0.07),DARK,rotation)
    return wall

# Main room is intentionally compact around the tapered core.
utility_wall('Utility_Front',(0,-1.62,5.55),(3.9,0.13,2.38),0,
             [(-0.92,5.42,0.78,1.82,'door'),(0.83,5.72,1.0,0.72,'window')])
utility_wall('Utility_Rear',(0,1.62,5.55),(3.9,0.13,2.38),PI,
             [(-0.85,5.42,0.78,1.82,'door'),(0.78,5.72,1.05,0.72,'window')])
utility_wall('Utility_Right',(1.89,0,5.55),(3.12,0.13,2.38),PI/2,
             [(-0.62,5.72,0.86,0.72,'window'),(0.68,5.72,0.68,0.72,'window')])
utility_wall('Utility_Left',(-1.89,0,5.55),(3.12,0.13,2.38),-PI/2,
             [(0.1,5.72,1.15,0.72,'window')])
box('Utility_Insulated_Roof',(0,0,6.79),(4.08,3.45,0.16),DARK)
# Vertical metal panel seams and lower reinforcement on all room faces.
for y,rot in ((-1.7,0),(1.7,PI)):
    for i in range(12):
        x = -1.8+i*0.33
        box('Utility_Cladding_Seam',(x,y,5.57),(0.018,0.022,2.3),RUST,rot)
for x,rot in ((-1.97,-PI/2),(1.97,PI/2)):
    for i in range(9):
        y = -1.4+i*0.35
        box('Utility_Cladding_Seam',(x,y,5.57),(0.018,0.022,2.3),RUST,rot)
# Left-side annex with a slightly sloped roof and louvred vent.
box('Annex_Floor',(-2.38,0.2,4.5),(1.22,2.7,0.13),DARK)
box('Annex_Left_Wall',(-2.94,0.2,5.35),(0.12,2.7,1.75),ORANGE)
box('Annex_Front_Wall',(-2.4,-1.1,5.35),(1.15,0.12,1.75),ORANGE)
box('Annex_Rear_Wall',(-2.4,1.5,5.35),(1.15,0.12,1.75),ORANGE)
box('Annex_Sloping_Roof',(-2.4,0.2,6.26),(1.35,2.9,0.12),ROOF,rotation=0.035)
for i in range(8):
    box('Annex_Vent_Louvre',(-2.4,-1.175,5.15+i*0.09),(0.7,0.07,0.035),DARK)
# Door signage, plates and utility markings.
box('Utility_Warning_Plate',(-0.9,-1.71,5.62),(0.28,0.018,0.32),LOGO_WHITE)
for i in range(3):
    box('Warning_Plate_Ink',(-0.9,-1.725,5.53+i*0.07),(0.17,0.012,0.018),SIGN_DARK)
# Roof-edge flashing and rainwater downpipes.
for y in (-1.75,1.75):
    rod('Utility_Gutter',(-2.04,y,6.79),(2.04,y,6.79),0.04,DARK)
for x,y in ((-1.96,1.72),(1.96,-1.72)):
    curve_path('Utility_Downpipe',[(x,y,6.78),(x,y,6.45),(x*1.04,y*1.04,6.2),(x*1.04,y*1.04,4.48)],0.035,RUST)
    for z in (4.8,5.7,6.4):
        cylinder('Downpipe_Clip',(x*1.04,y*1.04,z),0.052,0.045,DARK,12)


# -----------------------------------------------------------------------------
# D. Tapered shaft: individual corrugated panels, seams, bolts and inner rings.
# -----------------------------------------------------------------------------
print('BUILDING_TAPERED_SHAFT', flush=True)
use_collection('04_Tapered_Shaft')
Z_LEVELS = [4.42,6.85,10.55,14.98]
R_LEVELS = [2.12,1.98,1.55,1.18]
FACE_ANGLES = [-PI/2+i*PI/4 for i in range(8)]

def radius_at(z):
    for i in range(len(Z_LEVELS)-1):
        if Z_LEVELS[i] <= z <= Z_LEVELS[i+1]:
            f = (z-Z_LEVELS[i])/(Z_LEVELS[i+1]-Z_LEVELS[i])
            return R_LEVELS[i]*(1-f)+R_LEVELS[i+1]*f
    return R_LEVELS[0] if z < Z_LEVELS[0] else R_LEVELS[-1]


def corrugated_taper_panel(face_index, row):
    angle = FACE_ANGLES[face_index]
    z0,z1 = Z_LEVELS[row]+0.025,Z_LEVELS[row+1]-0.025
    r0,r1 = R_LEVELS[row],R_LEVELS[row+1]
    c0,n,t,h0 = face_frame(angle,r0,z0)
    c1,_,_,h1 = face_frame(angle,r1,z1)
    avg_width = h0+h1
    ribs = max(8,round(avg_width/0.145))
    nu = ribs*6
    nv = 5
    verts,uvs,faces = [],[],[]
    for j in range(nv+1):
        v = j/nv
        for i in range(nu+1):
            u = i/nu
            uu = 0.007 + u*0.986
            c = c0.lerp(c1,v)
            half = h0*(1-v)+h1*v
            # Trapezoidal-looking corrugation is actual displaced geometry.
            wave = math.cos(TAU*ribs*u) + 0.28*math.cos(2*TAU*ribs*u)
            p = c+t*((uu-0.5)*2*half)+n*(0.025+0.024*wave)
            verts.append(p)
            uvs.append((u*avg_width*0.48,v*(z1-z0)*0.48))
    for j in range(nv):
        for i in range(nu):
            a = j*(nu+1)+i
            faces.append((a,a+1,a+nu+2,a+nu+1))
    obj = mesh_object('Shaft_F%02d_R%02d_Corrugated_Panel'%(face_index,row),verts,faces,ORANGE,uvs,True)
    mod = obj.modifiers.new('Real sheet thickness','SOLIDIFY')
    mod.thickness = 0.045
    mod.offset = -1
    return obj

for face in range(8):
    for row in range(3):
        corrugated_taper_panel(face,row)
    angle = FACE_ANGLES[face]
    for zi,z in enumerate(Z_LEVELS):
        r = radius_at(z)
        c,n,t,h = face_frame(angle,r,z)
        # Horizontal reinforcement and visible fasteners.
        box('Shaft_Horizontal_Reinforcement',c+n*0.055,(2*h+0.035,0.075,0.105),RUST,
            math.atan2(t.y,t.x))
        for u in (-0.84,-0.42,0.0,0.42,0.84):
            add_bolt('Shaft_Seam_Bolt',c+t*(u*h)+n*0.108,n,0.019)
    # Rain streaks and narrow panel-edge strips, distributed on every face.
    for k in range(14):
        z = RNG.uniform(4.8,14.55)
        length = RNG.uniform(0.18,0.72)
        r0,r1 = radius_at(z),radius_at(min(14.9,z+length))
        c0,n,t,h0 = face_frame(angle,r0,z)
        c1,_,_,h1 = face_frame(angle,r1,min(14.9,z+length))
        u = RNG.uniform(-0.86,0.86)
        width = RNG.uniform(0.008,0.025)
        p0 = c0+t*(u*h0)+n*0.081
        p1 = c1+t*(u*h1)+n*0.081
        verts = [p0-t*width,p0+t*width,p1+t*width*0.35,p1-t*width*0.35]
        mesh_object('Vertical_Oxidation_Runoff',verts,[(0,1,2,3)],RUST,
                    [(0,0),(1,0),(1,1),(0,1)])
# Sloping corner posts and internal central support/rings.
for i in range(8):
    a = PI/8+i*PI/4
    for row in range(3):
        p0 = (R_LEVELS[row]*math.cos(a),R_LEVELS[row]*math.sin(a),Z_LEVELS[row])
        p1 = (R_LEVELS[row+1]*math.cos(a),R_LEVELS[row+1]*math.sin(a),Z_LEVELS[row+1])
        beam_between('Shaft_Corner_Angle',p0,p1,0.075,0.09,DARK)
use_collection('12_Internal_Structure')
rod('Internal_Central_Column',(0,0,4.3),(0,0,18.1),0.18,DARK,20)
for z,r in zip(Z_LEVELS,R_LEVELS):
    oct_ring('Internal_Ring_Frame',r*0.58,r*0.83,z+0.06,0.1,DARK)
    for i in range(8):
        a = TAU*i/8
        beam_between('Internal_Radial_Strut',(0,0,z),(r*0.75*math.cos(a),r*0.75*math.sin(a),z),0.055,0.08,DARK)
for z in (6.9,10.6,15.0):
    for y in (-0.85,0.85):
        beam_between('Control_and_Shaft_Internal_X',(-0.9,y,z-1.0),(0.9,y,z+1.0),0.05,0.07,DARK)

# Main ladder follows the changing taper rather than floating at one radius.
use_collection('10_Mechanical_Details')
ladder_x = 0.42
ladder_zs = [4.65+i*0.28 for i in range(38)]
for side in (-1,1):
    points = []
    for z in (4.55,6.85,10.55,14.95):
        r = radius_at(z)
        y = -r*math.cos(PI/8)-0.16
        points.append((ladder_x+side*0.24,y,z))
    curve_path('Main_Ladder_Stringer',points,0.028,DARK)
for z in ladder_zs:
    y = -radius_at(z)*math.cos(PI/8)-0.18
    rod('Main_Ladder_Rung',(ladder_x-0.25,y,z),(ladder_x+0.25,y,z),0.018,DARK,10)
for z in (5.0,6.8,8.5,10.4,12.3,14.3):
    y = -radius_at(z)*math.cos(PI/8)
    for side in (-1,1):
        rod('Main_Ladder_Standoff',(ladder_x+side*0.24,y,z),(ladder_x+side*0.24,y-0.24,z),0.022,DARK)
# Rear ladder provides coherent secondary access.
for x in (-0.23,0.23):
    curve_path('Rear_Ladder_Stringer',[(x,radius_at(z)*math.cos(PI/8)+0.14,z) for z in (6.8,10.55,14.95)],0.024,DARK)
for i in range(30):
    z = 6.85+i*0.27
    y = radius_at(z)*math.cos(PI/8)+0.15
    rod('Rear_Ladder_Rung',(-0.24,y,z),(0.24,y,z),0.017,DARK,10)

# Surface-mounted access hatches, windows and junction boxes around the tower.
use_collection('04_Tapered_Shaft')
for face in range(8):
    angle = FACE_ANGLES[face]
    for idx,z in enumerate((7.75,12.2)):
        r = radius_at(z)
        c,n,t,h = face_frame(angle,r,z)
        phi = math.atan2(t.y,t.x)
        offset = -0.34 if (face+idx)%2 else 0.28
        hc = c+t*offset+n*0.09
        box('Shaft_Access_Hatch_Frame',hc,(0.58,0.11,0.92),DARK,phi)
        box('Shaft_Access_Hatch',hc+n*0.065,(0.49,0.045,0.81),ORANGE,phi)
        for sx in (-1,1):
            for sz in (-1,1):
                add_bolt('Hatch_Fastener',hc+t*(sx*0.22)+Vector((0,0,sz*0.36))+n*0.105,n,0.016)
        rod('Hatch_Handle',hc+t*0.13+n*0.13+Vector((0,0,-0.08)),hc+t*0.13+n*0.13+Vector((0,0,0.08)),0.015,DARK)
    z = 9.15 if face%2 else 13.25
    c,n,t,h = face_frame(angle,radius_at(z),z)
    box('Electrical_Junction_Box',c+t*0.43+n*0.16,(0.32,0.2,0.46),BLUE,math.atan2(t.y,t.x))
    for dz in (-0.16,0.16):
        add_bolt('Electrical_Box_Bolt',c+t*0.43+n*0.27+Vector((0,0,dz)),n,0.015)
# Circular front instrument and its conduit.
z = 11.65
c,n,t,h = face_frame(-PI/2,radius_at(z),z)
cylinder('Round_Instrument_Rim',c+n*0.14-t*0.45,0.22,0.09,DARK,32,n.to_track_quat('Z','Y'))
cylinder('Round_Instrument_Dial',c+n*0.195-t*0.45,0.18,0.018,WHITE,32,n.to_track_quat('Z','Y'))
rod('Instrument_Pointer',c+n*0.213-t*0.45,c+n*0.213-t*0.39+Vector((0,0,0.1)),0.009,SIGN_DARK)
curve_path('Instrument_Conduit',[c-t*0.45+n*0.1,c-t*0.45+n*0.1-Vector((0,0,0.4)),c+n*0.1-Vector((0,0,0.75))],0.018,RUBBER)
# Pipework on three faces, with elbow curves and fixing collars.
for face in (1,2,5,7):
    a = FACE_ANGLES[face]
    pts = []
    for z in (4.55,6.85,10.55,14.8):
        c,n,t,h = face_frame(a,radius_at(z),z)
        pts.append(c-t*(h*0.66)+n*0.19)
    curve_path('External_Service_Riser',pts,0.04,RUST)
    for z in (5.2,7.1,9.6,12.0,14.1):
        c,n,t,h = face_frame(a,radius_at(z),z)
        p = c-t*(h*0.66)+n*0.19
        rod('Riser_Standoff',p-n*0.16,p+n*0.05,0.023,DARK)

# -----------------------------------------------------------------------------
# E. Wraparound maintenance platforms and exterior staircase.
# -----------------------------------------------------------------------------
print('BUILDING_SERVICE_PLATFORMS', flush=True)
use_collection('05_Service_Platforms')

def service_platform(name,z,outer,inner,rail_height=0.72):
    oct_ring(name+'_Floor',inner,outer,z,0.12,DARK)
    oct_ring(name+'_Outer_C Channel',outer-0.06,outer+0.035,z+0.02,0.19,RUST)
    railing_polygon(name+'_Guardrail',outer-0.025,z+0.03,rail_height)
    for i in range(8):
        a = PI/8+i*PI/4
        p0 = (inner*0.88*math.cos(a),inner*0.88*math.sin(a),z-0.72)
        p1 = ((outer-0.12)*math.cos(a),(outer-0.12)*math.sin(a),z-0.1)
        beam_between(name+'_Triangular_Bracket',p0,p1,0.065,0.09,DARK)
        # Radial floor bearer and grating strips.
        beam_between(name+'_Floor_Bearer',(inner*math.cos(a),inner*math.sin(a),z-0.08),
                     (outer*math.cos(a),outer*math.sin(a),z-0.08),0.05,0.08,DARK)
    return name

service_platform('Mid_Service_Platform',10.62,1.97,1.38,0.68)
service_platform('Lower_Service_Platform',7.12,2.47,1.8,0.72)
# Small left-side hatch landing, projecting farther than the main ring.
box('Mid_Landing_Extension',(-1.55,-0.35,10.6),(1.15,1.2,0.12),DARK)
for y in (-0.9,0.2):
    beam_between('Mid_Landing_Diagonal',(-1.15,y,9.92),(-2.0,y,10.54),0.065,0.08,DARK)
# Exterior staircase from lower deck to lower service platform.
start = Vector((-2.7,-1.91,4.48))
end = Vector((-1.63,-1.91,7.17))
path = end-start
flat = Vector((path.x,path.y,0)).normalized()
side = Vector((-flat.y,flat.x,0))
steps = 16
for i in range(steps):
    p = start.lerp(end,(i+0.5)/steps)
    box('Exterior_Stair_Tread',p,(0.68,0.25,0.065),DARK,math.atan2(flat.y,flat.x)-PI/2)
    # Anti-slip nosing.
    rod('Stair_Tread_Nosing',p-side*0.33+flat*0.11+Vector((0,0,0.04)),p+side*0.33+flat*0.11+Vector((0,0,0.04)),0.012,RUST)
for s in (-1,1):
    off = side*(s*0.36)
    beam_between('Stair_Stringer',start+off,end+off,0.075,0.12,DARK)
    for i in (0,4,8,12,15):
        p = start.lerp(end,i/15)+off
        rod('Stair_Railing_Post',p,p+Vector((0,0,0.78)),0.022,DARK)
    rod('Stair_Handrail',start+off+Vector((0,0,0.8)),end+off+Vector((0,0,0.8)),0.025,DARK)
    rod('Stair_Midrail',start+off+Vector((0,0,0.42)),end+off+Vector((0,0,0.42)),0.017,DARK)


# -----------------------------------------------------------------------------
# F. Elevated control room: thick walls, boolean openings, complete balcony.
# -----------------------------------------------------------------------------
print('BUILDING_CONTROL_ROOM', flush=True)
use_collection('06_Control_Room')
R_ROOM = 3.0
ROOM_BOTTOM,ROOM_TOP = 15.22,18.04
ROOM_MID = (ROOM_BOTTOM+ROOM_TOP)/2
# Tapered structural transition beneath the broad room.
verts,uvs,faces = [],[],[]
for r,z in ((1.2,14.82),(2.88,15.24)):
    for i in range(8):
        a = PI/8+i*PI/4
        verts.append((r*math.cos(a),r*math.sin(a),z))
        uvs.append((i/8*4,(z-14.82)*0.5))
for i in range(8):
    j=(i+1)%8
    faces.append((i,j,8+j,8+i))
transition = mesh_object('Control_Room_Structural_Transition',verts,faces,DARK,uvs)
mod = transition.modifiers.new('Transition plate thickness','SOLIDIFY')
mod.thickness = 0.1
cylinder('Control_Room_Floor',(0,0,15.24),2.87,0.16,DARK,8,(0,0,PI/8))
# Dark ceiling and a few interior consoles provide depth through real glazing.
cylinder('Control_Room_Ceiling',(0,0,17.96),2.8,0.1,DARK,8,(0,0,PI/8))
for x,y,rot in ((0,-1.4,0),(1.4,0.6,PI/2),(-1.1,0.8,0)):
    box('Interior_Control_Console',(x,y,15.95),(0.85,0.45,1.05),SIGN_DARK,rot)
    box('Interior_Dark_Monitor',(x,y-0.24,16.25),(0.55,0.035,0.36),GLASS,rot)

window_layout = {
    0: [],
    1: [(-0.48,16.25,0.62,0.72,1),(0.48,16.25,0.62,0.72,1),(-0.48,17.3,0.62,0.68,1),(0.48,17.3,0.62,0.68,1)],
    2: [(-0.48,16.55,0.72,1.15,1),(0.48,16.55,0.72,1.15,1)],
    3: [(-0.55,16.4,0.72,0.9,1),(0.55,16.4,0.72,0.9,1),(0,17.42,0.82,0.48,2)],
    4: [(-0.65,16.5,0.78,1.1,2),(0.45,16.5,0.98,1.1,2)],
    5: [(-0.48,16.25,0.62,0.72,1),(0.48,16.25,0.62,0.72,1),(-0.48,17.3,0.62,0.68,1),(0.48,17.3,0.62,0.68,1)],
    6: [(0,16.35,1.45,0.82,3)],
    7: [(0,16.05,1.48,0.72,3),(-0.64,17.32,0.36,0.64,0),(0,17.32,0.36,0.64,0),(0.64,17.32,0.36,0.64,0)],
}
for face,angle in enumerate(FACE_ANGLES):
    c,n,t,half = face_frame(angle,R_ROOM,ROOM_MID)
    phi = math.atan2(t.y,t.x)
    wall = box('Control_Room_Wall_%02d'%face,c,(2*half+0.018,0.16,ROOM_TOP-ROOM_BOTTOM),ORANGE,phi,unique=True)
    for wi,(x,z,w,h,mul) in enumerate(window_layout[face]):
        wc = Vector((c.x,c.y,z))+t*x
        cutter = box('Temporary_Window_Cutter',wc,(w,0.62,h),None,phi,unique=True)
        boolean_cut(wall,cutter)
        add_window('Control_F%02d_W%02d'%(face,wi),wc,n,t,w,h,mul)
    # Narrow folded-panel seams, avoiding glazed areas.
    for i in range(1,8):
        x = -half + 2*half*i/8
        blocked = any(abs(x-wx) < ww/2+0.04 for wx,wz,ww,wh,mul in window_layout[face])
        if not blocked:
            box('Control_Room_Panel_Seam',Vector((c.x,c.y,ROOM_MID))+t*x+n*0.091,
                (0.017,0.022,ROOM_TOP-ROOM_BOTTOM-0.06),RUST,phi)
    for z in (ROOM_BOTTOM+0.03,ROOM_TOP-0.02):
        box('Control_Room_Reinforcing_Beam',Vector((c.x,c.y,z))+n*0.045,
            (2*half+0.08,0.12,0.1),DARK,phi)
# Heavy corner posts and roof/balcony junction plates.
for i in range(8):
    a = PI/8+i*PI/4
    x,y = R_ROOM*math.cos(a),R_ROOM*math.sin(a)
    box('Control_Room_Corner_Post',(x,y,ROOM_MID),(0.11,0.11,ROOM_TOP-ROOM_BOTTOM+0.12),DARK,rotation=a)
    box('Control_Room_Corner_Cleat',(x,y,17.88),(0.24,0.2,0.12),RUST,rotation=a)
    add_bolt('Control_Corner_Bolt',(x,y-0.07,17.88),(0,-1,0),0.024)
# Wraparound balcony with actual thickness, toe boards and radial supports.
oct_ring('Control_Balcony_Floor',2.73,3.43,15.2,0.14,DARK)
oct_ring('Control_Balcony_Edge_Channel',3.36,3.47,15.23,0.24,RUST)
railing_polygon('Control_Balcony_Guardrail',3.41,15.24,0.74)
for i in range(16):
    a = PI/8+i*PI/8
    p0 = (1.62*math.cos(a),1.62*math.sin(a),14.45)
    p1 = (3.26*math.cos(a),3.26*math.sin(a),15.08)
    beam_between('Balcony_Knee_Bracket',p0,p1,0.07,0.1,DARK)
    if i%2 == 0:
        cylinder('Balcony_Connection_Pin',p1,0.065,0.16,RUST,12)
# Radial floor seams and anti-slip slats.
for i in range(32):
    a = TAU*i/32
    rod('Balcony_Floor_Seam',(2.82*math.cos(a),2.82*math.sin(a),15.215),
        (3.35*math.cos(a),3.35*math.sin(a),15.215),0.009,RUST,8)
# Short front roof-access ladder, visible beside the circular marking.
for x in (-1.03,-0.64):
    rod('Control_Roof_Ladder_Rail',(x,-2.92,15.26),(x,-2.92,18.47),0.025,DARK)
for i in range(12):
    z = 15.35+i*0.265
    rod('Control_Roof_Ladder_Rung',(-1.04,-2.94,z),(-0.63,-2.94,z),0.017,DARK)

# -----------------------------------------------------------------------------
# G. Reference markings: real mesh text and curved smile on the front wall.
# -----------------------------------------------------------------------------
use_collection('11_Markings')
LOGO_ORANGE = simple_material('Marking_Industrial_Orange','D85B17',0.68,0.05)
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT = bpy.data.fonts.load(font_path) if os.path.exists(font_path) else None

def text_object(name,body,location,size,material,rotation=(PI/2,0,0),font=None):
    data = bpy.data.curves.new(name+'_Text','FONT')
    data.body = body
    data.align_x = 'CENTER'
    data.align_y = 'CENTER'
    data.size = size
    data.space_character = 1.0
    data.resolution_u = 8
    if font or FONT:
        data.font = font or FONT
    data.materials.append(material)
    obj = link_object(bpy.data.objects.new(name,data))
    obj.location = location
    obj.rotation_euler = rotation
    return obj

front_c,front_n,front_t,front_half = face_frame(-PI/2,R_ROOM,17.22)
logo_center = front_c+front_n*0.14
cylinder('Large_White_Circular_Marking',logo_center,0.67,0.025,LOGO_WHITE,64,front_n.to_track_quat('Z','Y'))
# Stacked small lettering inside the circle.
text_object('Circular_Marking_FISH_1','FISH',logo_center+front_n*0.022+Vector((0,0,0.28)),0.19,LOGO_ORANGE)
text_object('Circular_Marking_FISH_2','FISH',logo_center+front_n*0.023+Vector((0,0,0.07)),0.19,LOGO_ORANGE)
text_object('Circular_Marking_BY','BY',logo_center+front_n*0.024+Vector((-0.19,0,-0.12)),0.17,LOGO_ORANGE)
# Abstract orange central mark and curved arrow smile.
curve_path('Circular_Marking_Central_Swirl',[
    logo_center+front_n*0.035+Vector((-0.02,0,0.08)),
    logo_center+front_n*0.035+Vector((0.13,0,0.02)),
    logo_center+front_n*0.035+Vector((0.12,0,-0.18)),
    logo_center+front_n*0.035+Vector((-0.02,0,-0.22)),
    logo_center+front_n*0.035+Vector((-0.08,0,-0.09))],0.045,LOGO_ORANGE)
smile_pts = []
for i in range(25):
    x = -0.43+0.86*i/24
    z = -0.27-0.16*(1-(x/0.43)**2)
    smile_pts.append(logo_center+front_n*0.04+Vector((x,0,z)))
curve_path('Circular_Marking_Smile',smile_pts,0.024,LOGO_ORANGE)
mesh_object('Circular_Marking_Arrowhead',[
    logo_center+front_n*0.045+Vector((0.32,0,-0.26)),
    logo_center+front_n*0.045+Vector((0.49,0,-0.24)),
    logo_center+front_n*0.045+Vector((0.42,0,-0.39))],[(0,1,2)],LOGO_ORANGE)
# Amazon wordmark and its own smile below the circle.
amazon_center = front_n*0.18+Vector((0,-R_ROOM*math.cos(PI/8),16.18))
text_object('Amazon_Wordmark','amazon',amazon_center,0.54,LOGO_WHITE)
word_smile = []
for i in range(25):
    x = -0.7+1.4*i/24
    z = -0.28-0.1*(1-(x/0.7)**2)
    word_smile.append(amazon_center+front_n*0.02+Vector((x,0,z)))
curve_path('Amazon_Smile',word_smile,0.018,LOGO_WHITE)
mesh_object('Amazon_Arrowhead',[
    amazon_center+front_n*0.025+Vector((0.56,0,-0.28)),
    amazon_center+front_n*0.025+Vector((0.76,0,-0.25)),
    amazon_center+front_n*0.025+Vector((0.68,0,-0.39))],[(0,1,2)],LOGO_WHITE)

# -----------------------------------------------------------------------------
# H. Hipped octagonal roof with actual corrugated faces, seams and gutters.
# -----------------------------------------------------------------------------
print('BUILDING_CORRUGATED_ROOF', flush=True)
use_collection('07_Roof')
R_EAVE,R_CROWN = 3.68,1.28
Z_EAVE,Z_CROWN = 18.11,19.28
for face in range(8):
    a0 = PI/8+face*PI/4
    a1 = a0+PI/4
    outer0 = Vector((R_EAVE*math.cos(a0),R_EAVE*math.sin(a0),Z_EAVE))
    outer1 = Vector((R_EAVE*math.cos(a1),R_EAVE*math.sin(a1),Z_EAVE))
    inner0 = Vector((R_CROWN*math.cos(a0),R_CROWN*math.sin(a0),Z_CROWN))
    inner1 = Vector((R_CROWN*math.cos(a1),R_CROWN*math.sin(a1),Z_CROWN))
    normal = (outer1-outer0).cross(inner0-outer0).normalized()
    if normal.z < 0:
        normal = -normal
    nu,nv = 72,10
    verts,uvs,faces = [],[],[]
    for j in range(nv+1):
        v = j/nv
        for i in range(nu+1):
            u = i/nu
            outer = outer0.lerp(outer1,u)
            inner = inner0.lerp(inner1,u)
            wave = 0.027*(math.cos(TAU*12*u)+0.25*math.cos(TAU*24*u))
            p = outer.lerp(inner,v)+normal*wave
            verts.append(p)
            uvs.append((u*2.4,v*2.8))
    for j in range(nv):
        for i in range(nu):
            a = j*(nu+1)+i
            faces.append((a,a+1,a+nu+2,a+nu+1))
    obj = mesh_object('Roof_Corrugated_Hip_%02d'%face,verts,faces,ROOF,uvs,True)
    mod = obj.modifiers.new('Roof sheet thickness','SOLIDIFY')
    mod.thickness = 0.035
    # Raised standing seams, eave gutters and under-roof support.
    rod('Roof_Radial_Standing_Seam',outer0+normal*0.035,inner0+normal*0.035,0.025,ROOF,10)
    rod('Roof_Eave_Gutter',outer0,outer1,0.047,DARK,12)
    beam_between('Roof_Underside_Rafter',(0,0,18.0),outer0-Vector((0,0,0.08)),0.07,0.1,DARK)
# Raised crown, flashing and small ventilator curb.
cylinder('Roof_Raised_Crown',(0,0,Z_CROWN),R_CROWN,0.1,ROOF,8,(0,0,PI/8))
oct_ring('Roof_Crown_Flashing',1.22,1.39,Z_CROWN+0.025,0.08,DARK)
box('Roof_Access_Curb',(0.55,0.25,19.38),(0.64,0.72,0.18),DARK)
box('Roof_Access_Hatch',(0.55,0.25,19.5),(0.58,0.66,0.06),ROOF)
# Eave underside is finished rather than open.
cylinder('Roof_Dark_Underside',(0,0,18.04),2.87,0.08,DARK,8,(0,0,PI/8))


# -----------------------------------------------------------------------------
# I. Communications: concave dishes, masts, guys, chimney and spherical sensor.
# -----------------------------------------------------------------------------
print('BUILDING_COMMUNICATIONS', flush=True)
use_collection('08_Communications')

def roof_height(x,y):
    r = math.hypot(x,y)
    if r <= R_CROWN:
        return Z_CROWN+0.08
    return Z_CROWN-(r-R_CROWN)/(R_EAVE-R_CROWN)*(Z_CROWN-Z_EAVE)+0.055


def satellite_dish(name,center,direction,radius,base):
    before = set(bpy.data.objects)
    root = link_object(bpy.data.objects.new(name+'_Aim_Assembly',None))
    root.empty_display_type = 'PLAIN_AXES'
    root.empty_display_size = 0.2
    depth = radius*0.28
    nr,ns = 12,48
    verts = [(0,0,0)]
    uvs = [(0.5,0.5)]
    for j in range(1,nr+1):
        rr = radius*j/nr
        zz = depth*(rr/radius)**2
        for i in range(ns):
            a = TAU*i/ns
            verts.append((rr*math.cos(a),rr*math.sin(a),zz))
            uvs.append((0.5+0.5*rr/radius*math.cos(a),0.5+0.5*rr/radius*math.sin(a)))
    faces = []
    for i in range(ns):
        faces.append((0,1+i,1+(i+1)%ns))
    for j in range(nr-1):
        s0,s1 = 1+j*ns,1+(j+1)*ns
        for i in range(ns):
            ni=(i+1)%ns
            faces.append((s0+i,s1+i,s1+ni,s0+ni))
    dish = mesh_object(name+'_Concave_Reflector',verts,faces,WHITE,uvs,True)
    mod = dish.modifiers.new('Reflector shell thickness','SOLIDIFY')
    mod.thickness = 0.014
    rim = [(radius*math.cos(TAU*i/64),radius*math.sin(TAU*i/64),depth) for i in range(64)]
    curve_path(name+'_Rolled_Rim',rim,0.014,WHITE,True)
    cylinder(name+'_Rear_Hub',(0,0,-0.07),radius*0.14,0.15,DARK,24)
    for i in range(8):
        a=TAU*i/8
        rod(name+'_Back_Radial_Rib',(0,0,-0.04),(radius*0.91*math.cos(a),radius*0.91*math.sin(a),depth*0.78),0.012,DARK,8)
    focal = radius*0.85
    for a in (PI/2,PI*7/6,PI*11/6):
        rod(name+'_Receiver_Support',(radius*0.92*math.cos(a),radius*0.92*math.sin(a),depth),
            (0,0,focal),0.012,DARK,8)
    cylinder(name+'_Central_Receiver',(0,0,focal),radius*0.09,radius*0.22,WHITE,16)
    curve_path(name+'_Receiver_Cable',[(0,0,focal),(0.04,0.02,focal*0.6),(0.08,0.03,depth),(0.06,0,-0.14)],0.008,RUBBER)
    # Small rear elevation bracket and pivot.
    box(name+'_Elevation_Bracket',(0,0,-0.17),(radius*0.28,radius*0.22,0.16),DARK)
    for obj in set(bpy.data.objects)-before:
        if obj != root:
            obj.parent = root
    direction = Vector(direction).normalized()
    root.rotation_mode = 'QUATERNION'
    root.rotation_quaternion = direction.to_track_quat('Z','Y')
    root.location = center
    # World-vertical mount stays independent of dish tilt.
    rod(name+'_Mounting_Mast',base,Vector(center)-direction*0.12,0.045,DARK,16)
    sphere(name+'_Azimuth_Pivot',Vector(center)-direction*0.14,0.1,DARK)
    box(name+'_Roof_Baseplate',base,(0.36,0.36,0.07),RUST)
    for sx in (-1,1):
        for sy in (-1,1):
            add_bolt(name+'_Base_Anchor',Vector(base)+Vector((sx*0.12,sy*0.12,0.05)),(0,0,1),0.018)
    return root

satellite_dish('Large_Roof_Dish',(-2.03,-0.15,19.72),(-0.62,-0.74,0.32),0.57,(-2.03,-0.15,roof_height(-2.03,-0.15)))
satellite_dish('Front_Balcony_Dish',(-1.25,-3.13,15.82),(0.08,-0.97,0.24),0.38,(-1.25,-3.05,15.22))
satellite_dish('Rear_Communications_Dish',(1.15,1.22,19.73),(0.45,0.83,0.34),0.38,(1.15,1.22,roof_height(1.15,1.22)))


def communication_mast(name,base,height,main=False):
    x,y,z = base
    cylinder(name+'_Base_Flange',(x,y,z+0.035),0.14,0.07,RUST,20)
    rod(name+'_Lower_Mast',(x,y,z),(x,y,z+height*0.58),0.035,DARK,12)
    rod(name+'_Upper_Mast',(x,y,z+height*0.58),(x,y,z+height),0.022,DARK,10)
    for dz in (0.2,height*0.55,height*0.78):
        cylinder(name+'_Mast_Collar',(x,y,z+dz),0.052,0.075,RUST,12)
    # Two crossarms with ceramic insulators and terminal fittings.
    for frac,span in ((0.64,0.68),(0.84,0.46)):
        zz=z+height*frac
        rod(name+'_Crossarm',(x-span/2,y,zz),(x+span/2,y,zz),0.018,DARK,10)
        for sx in (-1,1):
            cylinder(name+'_Insulator',(x+sx*span*0.38,y,zz+0.055),0.027,0.1,WHITE,10)
    # Yagi/directional elements at the top.
    top=z+height*0.91
    rod(name+'_Yagi_Boom',(x-0.5,y,top),(x+0.5,y,top),0.014,DARK,8)
    for i in range(6):
        xx=x-0.43+i*0.172
        span=0.5-0.035*i
        rod(name+'_Yagi_Element',(xx,y-span/2,top),(xx,y+span/2,top),0.009,DARK,8)
    rod(name+'_Top_Finial',(x,y,z+height-0.1),(x,y,z+height+0.25),0.01,DARK,8)
    if main:
        sphere(name+'_Beacon_Base',(x,y,z+height+0.05),0.055,DARK)
        sphere(name+'_Red_Beacon',(x,y,z+height+0.14),0.038,RED,scale=(1,1,1.35))
    for i in range(3):
        a=TAU*i/3+0.25
        anchor=(x+0.95*math.cos(a),y+0.95*math.sin(a),z+0.02)
        sag_wire(name+'_Guy_Wire',(x,y,z+height*0.66),anchor,0.07,0.006,DARK,20)
        cylinder(name+'_Guy_Anchor',anchor,0.045,0.1,RUST,10)
    return (x,y,z+height)

communication_mast('Primary_Antenna',(-0.15,0.08,19.39),3.45,True)
communication_mast('Secondary_Antenna',(0.63,0.35,19.39),2.85)
communication_mast('Rear_Antenna',(-0.58,0.68,19.39),2.15)
communication_mast('Roof_Edge_Antenna',(1.68,-0.52,roof_height(1.68,-0.52)),1.5)
# Roof chimney/exhaust with collars, rain cap and heat shield.
x,y=-1.18,0.55
z=roof_height(x,y)
cylinder('Exhaust_Roof_Flashing',(x,y,z+0.03),0.25,0.08,RUST,24)
rod('Ventilation_Exhaust',(x,y,z),(x,y,z+1.13),0.12,DARK,24)
for dz in (0.15,0.48,0.9):
    cylinder('Exhaust_Collar',(x,y,z+dz),0.15,0.08,RUST,24)
cylinder('Exhaust_Rain_Cap',(x,y,z+1.18),0.19,0.07,DARK,24)
for sx in (-1,1):
    rod('Exhaust_Cap_Spacer',(x+sx*0.09,y,z+1.04),(x+sx*0.09,y,z+1.17),0.015,DARK)
# Small radar-like equipment and junction housings.
for x,y in ((0.8,-0.72),(-0.75,-0.58),(0.2,0.95)):
    z=roof_height(x,y)
    box('Roof_Radio_Junction',(x,y,z+0.14),(0.32,0.26,0.25),BLUE)
    cylinder('Roof_Radio_Connector',(x,y,z+0.34),0.035,0.18,WHITE,12)
# Spherical sensor on a triangulated metal stand, with an actual protruding lens.
sensor = Vector((2.28,0.35,19.78))
base_z = roof_height(2.28,0.35)
for i in range(3):
    a=TAU*i/3
    foot=Vector((2.28+0.35*math.cos(a),0.35+0.35*math.sin(a),base_z))
    rod('Sensor_Tripod_Leg',foot,sensor+Vector((0,0,-0.28)),0.027,DARK,12)
    box('Sensor_Tripod_Foot',foot,(0.18,0.16,0.05),RUST)
rod('Sensor_Central_Stand',(2.28,0.35,base_z),(2.28,0.35,19.53),0.045,DARK)
sphere('Spherical_Sensor_Housing',sensor,0.39,DARK,scale=(1,1,1.08))
lens_dir=Vector((0.4,-0.9,0.12)).normalized()
cylinder('Sensor_Lens_Barrel',sensor+lens_dir*0.34,0.205,0.18,RUST,32,lens_dir.to_track_quat('Z','Y'))
cylinder('Sensor_Lens_Rim',sensor+lens_dir*0.44,0.17,0.06,WHITE,32,lens_dir.to_track_quat('Z','Y'))
cylinder('Sensor_Optical_Glass',sensor+lens_dir*0.478,0.135,0.025,GLASS_2,32,lens_dir.to_track_quat('Z','Y'))
sphere('Sensor_Lens_Highlight',sensor+lens_dir*0.495+Vector((-0.025,0,0.035)),0.055,GLASS)
cylinder('Sensor_Top_Cap',sensor+Vector((0,0,0.4)),0.11,0.09,RUST,20)

# -----------------------------------------------------------------------------
# J. Left-side industrial boom: paired trusses, mounts, braces and hanging lines.
# -----------------------------------------------------------------------------
print('BUILDING_SIDE_BOOM', flush=True)
use_collection('09_Crane')
for y in (-0.55,0.12):
    for z in (16.65,17.13):
        beam_between('Crane_Longitudinal_Chord',(-2.38,y,z),(-6.62,y,z),0.085,0.11,DARK)
    for i in range(7):
        x=-2.42-i*0.6
        beam_between('Crane_Web_Brace',(x,y,16.66),(x-0.58,y,17.12),0.045,0.055,RUST)
        beam_between('Crane_Web_Brace',(x,y,17.12),(x-0.58,y,16.66),0.035,0.045,DARK)
    beam_between('Crane_Underside_Knee',(-2.36,y,15.56),(-4.22,y,16.65),0.085,0.11,DARK)
    box('Crane_Root_Mounting_Plate',(-2.43,y,16.87),(0.16,0.32,0.78),RUST)
    for zz in (16.62,17.12):
        cylinder('Crane_Root_Pivot',(-2.5,y,zz),0.1,0.26,DARK,20,(PI/2,0,0))
for i in range(8):
    x=-2.42-i*0.6
    beam_between('Crane_Transverse_Tie',(x,-0.57,16.68),(x,0.14,16.68),0.05,0.07,DARK)
    if i%2==0:
        beam_between('Crane_Top_Transverse_Tie',(x,-0.57,17.13),(x,0.14,17.13),0.04,0.05,DARK)
# Terminal fittings, small vertical rod, pulley and hook block.
box('Crane_Terminal_Crosshead',(-6.65,-0.22,16.93),(0.14,0.98,0.62),DARK)
rod('Crane_Terminal_Antenna',(-6.68,-0.2,17.12),(-6.68,-0.2,18.05),0.014,DARK,8)
rod('Crane_Terminal_Top_Crossbar',(-6.9,-0.2,17.86),(-6.47,-0.2,17.86),0.012,DARK,8)
cylinder('Crane_Pulley',(-6.64,-0.24,16.57),0.16,0.12,RUST,24,(PI/2,0,0))
sag_wire('Crane_Hanging_Load_Line',(-6.63,-0.25,16.48),(-6.58,-0.25,14.96),0.03,0.012,RUBBER)
box('Crane_Hook_Block',(-6.58,-0.25,15.0),(0.14,0.12,0.22),DARK)
curve_path('Crane_Hook',[(-6.58,-0.25,14.92),(-6.58,-0.25,14.8),(-6.48,-0.25,14.75),(-6.41,-0.25,14.83)],0.02,DARK)
for y in (-0.58,0.15):
    sag_wire('Crane_Looping_Power_Cable',(-6.5,y,16.55),(-2.46,y,16.14),0.78,0.014,RUBBER,36)
sag_wire('Crane_Control_Cable',(-2.42,-0.62,17.16),(-5.4,-0.62,17.14),0.18,0.009,RUBBER)

# -----------------------------------------------------------------------------
# K. Additional machinery and storage on every side of the lower deck.
# -----------------------------------------------------------------------------
use_collection('10_Mechanical_Details')
# Horizontal blue-gray tank on the right, including saddles and end caps.
rod('Deck_Water_Tank',(2.32,-0.15,4.88),(2.32,0.95,4.88),0.31,BLUE,32)
for y in (-0.15,0.95):
    sphere('Tank_Dished_End',(2.32,y,4.88),0.31,BLUE,scale=(1,0.22,1))
for y in (0.0,0.72):
    box('Tank_Saddle',(2.32,y,4.54),(0.75,0.13,0.24),DARK)
curve_path('Tank_Feed_Pipe',[(2.32,0.9,4.85),(2.65,1.1,4.75),(2.7,1.25,4.5),(1.8,1.45,4.48)],0.032,RUST)
cylinder('Tank_Top_Valve',(2.32,0.35,5.24),0.05,0.17,RUST,16)
rod('Tank_Valve_Handwheel',(2.19,0.35,5.34),(2.45,0.35,5.34),0.018,DARK)
# Cabinets and louvres at front and rear.
for x,y,rot in ((1.45,-1.95,0),(-1.2,1.96,PI),(2.72,-1.15,PI/2)):
    box('Deck_Electrical_Cabinet',(x,y,4.93),(0.56,0.32,0.96),BLUE,rot)
    for i in range(5):
        box('Cabinet_Vent_Slat',(x,y-0.17,4.72+i*0.065),(0.35,0.025,0.02),DARK,rot)
    cylinder('Cabinet_Latch',(x+0.18,y-0.18,5.08),0.025,0.045,DARK,12,(PI/2,0,0))
# Small storage drums and a cable reel.
for x,y in ((-2.55,1.85),(-2.15,1.88),(2.55,1.75)):
    cylinder('Deck_Storage_Drum',(x,y,4.7),0.18,0.52,RUST,20)
    for z in (4.5,4.88):
        cylinder('Drum_Rolled_Rim',(x,y,z),0.195,0.035,DARK,20)
cylinder('Cable_Reel_Core',(-2.55,-1.65,4.7),0.16,0.45,RUBBER,24,(0,PI/2,0))
for x in (-2.79,-2.31):
    cylinder('Cable_Reel_Flange',(x,-1.65,4.7),0.3,0.045,DARK,24,(0,PI/2,0))
# Coiled hose on rear railing.
coil=[]
for i in range(150):
    a=TAU*5*i/149
    coil.append((0.9+0.29*math.cos(a),2.26,5.05+0.29*math.sin(a)))
curve_path('Coiled_Service_Hose',coil,0.018,RUBBER)
# Small deck floodlights and their brackets (not emissive in overcast daylight).
for x,y in ((-2.85,-2.15),(2.85,2.15),(-2.85,2.15)):
    rod('Deck_Light_Post',(x,y,4.46),(x,y,5.35),0.025,DARK)
    box('Deck_Light_Housing',(x,y,5.37),(0.26,0.16,0.16),DARK)
    box('Deck_Light_Glass',(x,y-0.09,5.37),(0.2,0.018,0.1),WHITE)


# -----------------------------------------------------------------------------
# L. Convert curves/text, export a self-contained GLB, and render inspection views.
# -----------------------------------------------------------------------------
print('FINALIZING_MODEL', flush=True)
model_collections = [c for name,c in COLLECTIONS.items() if not name.startswith('90_')]
model_objects = []
for coll in model_collections:
    model_objects.extend(list(coll.objects))
# Curves remain editable throughout generation; mesh conversion makes them glTF-safe.
convertible = [o for o in model_objects if o.type in {'CURVE','FONT'}]
if convertible:
    bpy.ops.object.select_all(action='DESELECT')
    for obj in convertible:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = convertible[0]
    bpy.ops.object.convert(target='MESH')
# Recalculate mesh validation and attach useful assembly metadata.
model_objects = []
object_collection = {}
for coll in model_collections:
    for obj in coll.objects:
        if obj.type == 'MESH':
            obj.data.validate(verbose=False)
            obj['procedural_assembly'] = coll.name
            obj['units'] = 'meters'
            model_objects.append(obj)
            object_collection[obj] = coll

# Real-time export batch: evaluate every modifier in world space, then join only
# within the same architectural assembly and material. The editable source parts
# remain in the Blender scene; the GLB gets a compact assembly/material hierarchy.
print('BATCHING_REALTIME_EXPORT',flush=True)
export_collection = bpy.data.collections.new('13_Optimized_GLTF_Export')
SCENE.collection.children.link(export_collection)
export_collection.hide_render = True
depsgraph = bpy.context.evaluated_depsgraph_get()
groups = {}
for obj in model_objects:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated,preserve_all_data_layers=True,depsgraph=depsgraph)
    material = obj.material_slots[0].material if obj.material_slots else None
    if material:
        mesh.materials.clear()
        mesh.materials.append(material)
    duplicate = bpy.data.objects.new(obj.name+'_Evaluated',mesh)
    export_collection.objects.link(duplicate)
    duplicate.matrix_world = obj.matrix_world.copy()
    key = (object_collection[obj].name,material.name if material else 'Unassigned')
    groups.setdefault(key,[]).append(duplicate)
optimized_objects = []
for (assembly,material_name),objects in groups.items():
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects)>1:
        bpy.ops.object.join()
    merged = bpy.context.view_layer.objects.active
    merged.name = assembly.split('_',1)[-1]+'__'+material_name
    merged['procedural_assembly'] = assembly
    merged['source_part_count'] = len(objects)
    optimized_objects.append(merged)

bpy.ops.object.select_all(action='DESELECT')
for obj in optimized_objects:
    obj.select_set(True)
if optimized_objects:
    bpy.context.view_layer.objects.active = optimized_objects[0]
GLB_PATH = OUT / 'Offshore_Observation_Tower.glb'
print('EXPORTING_GLB',GLB_PATH,flush=True)
bpy.ops.export_scene.gltf(
    filepath=str(GLB_PATH),
    export_format='GLB',
    use_selection=True,
    export_apply=True,
    export_materials='EXPORT',
    export_cameras=False,
    export_lights=False,
    export_animations=False,
    export_extras=True,
    export_yup=True,
)
base_triangles = sum(sum(len(p.vertices)-2 for p in o.data.polygons) for o in model_objects)
optimized_triangles = sum(sum(len(p.vertices)-2 for p in o.data.polygons) for o in optimized_objects)
manifest = {
    'asset': 'Offshore Observation Tower',
    'generator': Path(__file__).name,
    'blender_version': bpy.app.version_string,
    'units': 'meters',
    'height_m': 23.2,
    'source_mesh_objects': len(model_objects),
    'optimized_draw_objects': len(optimized_objects),
    'base_mesh_triangles_before_export_modifiers': base_triangles,
    'optimized_export_triangles': optimized_triangles,
    'glb_bytes': GLB_PATH.stat().st_size,
    'collections': [c.name for c in model_collections],
    'textures': 'Baked base color, ORM and tangent normal; packed into GLB',
}
(OUT/'Tower_Asset_Manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('ASSET_MANIFEST',json.dumps(manifest),flush=True)

if opts.render:
    use_collection('90_Presentation')
    ground_mat = simple_material('Presentation_Muted_Ground','B0B4AC',0.96,0.0)
    ground = box('Presentation_Ground',(0,0,-0.13),(2000,2000,0.12),ground_mat)
    world = bpy.data.worlds.new('Overcast_Gray_Green_Sky')
    SCENE.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    bg.inputs['Color'].default_value = (0.48,0.52,0.49,1)
    bg.inputs['Strength'].default_value = 1.0

    def area_light(name,location,energy,size,color):
        data=bpy.data.lights.new(name,'AREA')
        data.energy=energy
        data.shape='DISK'
        data.size=size
        data.color=color
        obj=link_object(bpy.data.objects.new(name,data))
        obj.location=location
        obj.rotation_euler=(Vector((0,0,10))-obj.location).to_track_quat('-Z','Y').to_euler()
        return obj

    area_light('Large_Overcast_Key',(-12,-16,29),3000,18,(1.0,0.91,0.8))
    area_light('Sky_Fill',(14,6,24),1700,16,(0.78,0.88,1.0))
    area_light('Soft_Frontal_Fill',(2,-22,12),650,12,(0.92,0.96,1.0))
    camera_data=bpy.data.cameras.new('Inspection_Camera')
    camera=link_object(bpy.data.objects.new('Inspection_Camera',camera_data))
    SCENE.camera=camera
    camera_data.type='ORTHO'
    camera_data.ortho_scale=28.0
    camera_data.lens=50
    camera_data.clip_end=300
    SCENE.render.engine='CYCLES'
    SCENE.cycles.samples=opts.samples
    SCENE.cycles.use_denoising=True
    SCENE.render.image_settings.file_format='PNG'
    SCENE.render.image_settings.color_mode='RGBA'
    SCENE.render.film_transparent=False
    SCENE.view_settings.view_transform='AgX'
    try:
        SCENE.view_settings.look='AgX - Medium High Contrast'
    except Exception:
        pass
    SCENE.view_settings.exposure=0.65

    def render_view(filename,position,target,resolution=(800,1000),scale=28.0,hide_ground=False):
        ground.hide_render=hide_ground
        camera.location=position
        camera.rotation_euler=(Vector(target)-camera.location).to_track_quat('-Z','Y').to_euler()
        camera_data.ortho_scale=scale
        SCENE.render.resolution_x=resolution[0]
        SCENE.render.resolution_y=resolution[1]
        SCENE.render.resolution_percentage=100
        SCENE.render.filepath=str(filename)
        print('RENDERING',filename,flush=True)
        bpy.ops.render.render(write_still=True)

    render_view(OUT/'Tower_Reference_View.png',(23,-38,20),(0,0,11.25),(900,1200),27.8)
    inspection = [
        ('01_Rear.png',(-23,38,19),(0,0,11.2)),
        ('02_Left.png',(-38,-8,18),(0,0,11.2)),
        ('03_Right.png',(38,8,18),(0,0,11.2)),
        ('04_Above.png',(10,-14,40),(0,0,11.0)),
        ('05_Below.png',(18,-26,-7),(0,0,10.0)),
    ]
    for name,pos,target in inspection:
        render_view(VIEWS/name,pos,target,(760,900),28.5,name=='05_Below.png')
    ground.hide_render=False

print('TOWER_GENERATION_COMPLETE',flush=True)
