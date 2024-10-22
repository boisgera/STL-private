import marimo

__generated_with = "0.9.11"
app = marimo.App()


@app.cell
def __(mo):
    mo.md(
        r"""
        # STL

        "Stereolitography Language"... or "Standard Triangle Language"
        """
    )
    return


@app.cell
def __(show):
    show("data/teapot.stl", theta=60.0, phi=30.0, scale=2.0)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Simple STL files (Discovery)

        **TODO : cube & "eroded" cube.

        - [ ] Facet
        - [ ] Simplex
        - [ ] Cube
        - [ ] (Eroded Cube)
        - [ ] Pyramid
        """
    )
    return


@app.cell
def __():
    facet = """
    facet normal {n[0]} {n[1]} {n[2]}
        outer loop
            vertex {t[0][0]} {t[0][1]} {t[0][2]}
            vertex {t[1][0]} {t[1][1]} {t[1][2]}
            vertex {t[2][0]} {t[2][1]} {t[2][2]}
        endloop
    endfacet
    """
    return (facet,)


@app.cell
def __(np):
    normal = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    normal = normal / np.linalg.norm(normal)
    normal
    return (normal,)


@app.cell
def __(np):
    triangle = np.array([
        [1.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0], 
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return (triangle,)


@app.cell
def __(facet, normal, triangle):
    print(facet.format(n=normal, t=triangle))
    return


@app.cell
def __(facet, normal, triangle):
    name = "simplex"

    solid = f"""
    solid {name}

    {facet.format(n=normal, t=triangle)}

    endsolid {name}
    """

    print(solid)
    return name, solid


@app.cell
def __(name, solid):
    with open(f'output/{name}.stl', mode='wt') as _file:
        _file.write(solid)
    return


@app.cell
def __(facet, normal, triangle):
    name_1 = 'simplex'
    solid_1 = f'\nsolid {name_1}\n\n{facet.format(n=-normal, t=triangle)}\n\nendsolid {name_1}\n'
    print(solid_1)
    return name_1, solid_1


@app.cell
def __(solid_1):
    name_2 = 'simplex-opposite-normal'
    with open(f'output/{name_2}.stl', mode='wt') as _file:
        _file.write(solid_1)
    return (name_2,)


@app.cell
def __(facet, np, triangle):
    normal_1 = np.array([12.0, -56.0, 456.0], dtype=np.float32)
    name_3 = 'simplex'
    solid_2 = f'\nsolid {name_3}\n\n{facet.format(n=normal_1, t=triangle)}\n\nendsolid {name_3}\n'
    print(solid_2)
    name_3 = 'simplex-random-normal'
    with open(f'output/{name_3}.stl', mode='wt') as _file:
        _file.write(solid_2)
    return name_3, normal_1, solid_2


@app.cell
def __(facet, np, triangle):
    normal_2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    name_4 = 'simplex'
    solid_3 = f'\nsolid {name_4}\n\n{facet.format(n=normal_2, t=triangle)}\n\nendsolid {name_4}\n'
    print(solid_3)
    name_4 = 'simplex-zero-normal'
    with open(f'output/{name_4}.stl', mode='wt') as _file:
        _file.write(solid_3)
    return name_4, normal_2, solid_3


@app.cell
def __(mo):
    mo.md(
        r"""
        **TODO** warning about the behavior of the readers wrt normals and outer/inner representation.
        Inner may be displayed as outer or not at all and the normal may not be used.

        AFAICT the GitHub viewer will:

        - discard the normal info when it comes to determine the normal (use orientation instead)

        - only display the outer face

        - BUT, display this face is black when the normal in the file does not match its computation.

        YMML
        """
    )
    return


@app.cell
def __(facet, np):
    normal_3 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    normal_3 = normal_3 / np.linalg.norm(normal_3)
    triangle_1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    with open('output/pyramid.stl', mode='tw') as _file:
        _file.write('solid pyramid\n\n')
        for _i in [0, 1, 2, 3]:
            _file.write(facet.format(n=normal_3, t=triangle_1))
            normal_3 = R @ normal_3
            triangle_1 = triangle_1 @ R.T
        _file.write('endsolid pyramid\n')
    return R, normal_3, triangle_1


@app.cell
def __(mo):
    mo.md(r"""## NumPy to STL""")
    return


@app.cell
def __(np):
    # triangles -> faces or facets?
    # make a file-based output?

    def make_STL(triangles, normals=None, name=''):
        triangles = np.array(triangles, dtype=np.float32)
        if normals is None:
            d1 = triangles[:, 1, :] - triangles[:, 0, :]
            d2 = triangles[:, 2, :] - triangles[:, 1, :]
            vector_product = np.linalg.cross(d1, d2)
            norms = np.linalg.norm(vector_product, axis=1)
            normals = np.diag(1 / norms) @ vector_product
        parts = []
        parts.append(f"solid {name}\n")
        for t, n in zip(triangles, normals):
            parts.append(f"""facet normal {n[0]} {n[1]} {n[2]}
            outer loop
                vertex {t[0][0]} {t[0][1]} {t[0][2]}
                vertex {t[1][0]} {t[1][1]} {t[1][2]}
                vertex {t[2][0]} {t[2][1]} {t[2][2]}
            endloop
        endfacet
        """)
        parts.append(f"endsolid {name}\n")
        return "".join(parts)
    return (make_STL,)


@app.cell
def __(make_STL, np):
    _normals = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    _normals = _normals / np.linalg.norm(_normals)
    _normals = np.array([_normals], dtype=np.float32)
    triangles = np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
    out = make_STL(triangles, _normals, name='simplex')
    print(out)
    return out, triangles


@app.cell
def __(make_STL, triangles):
    make_STL(triangles)
    return


@app.cell
def __(mo):
    mo.md(r"""## STL to NumPy""")
    return


@app.cell
def __(np):
    STL_KEYWORDS = [
        'solid', 'endsolid', 
        'facet', 'endfacet', 
        'outer', 'loop', 'endloop', 
        'vertex'
    ]

    def tokenize(STL_text):
        raw_tokens = STL_text.split()
        tokens = []
        for token in raw_tokens:
            try: # is the token a number?
                tokens.append(np.float32(token))
            except ValueError:
                if token in STL_KEYWORDS:
                    tokens.append(token)
        return tokens
    return STL_KEYWORDS, tokenize


@app.cell
def __(np):
    def parse(tokens):
        normals = []
        for i, token in enumerate(tokens):
            if token == "facet":
                normals.append(np.array(tokens[i+1:i+4]))
        triangles = []
        for i, token in enumerate(tokens):
            if token == "loop":
                triangle = []
                triangle.append(np.array(tokens[i+2:i+5]))
                triangle.append(np.array(tokens[i+6:i+9]))
                triangle.append(np.array(tokens[i+10:i+13]))
                triangles.append(np.array(triangle))
        return np.array(triangles), np.array(normals)
    return (parse,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Diagnostics

        Make a diagnostic dataframe that checks for:

          - [ ] orientation rule (violations to the rhs rule)
          - [ ] vertex rule (every triangle should share two vertices with the adjacent triangles)
          - [ ] All positive octant rule
          - [ ] Triangle sorting rule (ascending z)
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""## STL Viewer""")
    return


@app.cell
def __(Camera, Mesh, glm, meshio, plt):
    def show(
        filename,
        theta=0.0,
        phi=0.0,
        scale=1.0,
        colormap="viridis",
        edgecolors=(0, 0, 0, 0.25),
        figsize=(6, 5),
    ):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, +1], ylim=[-1, +1], aspect=1)
        ax.axis("off")
        camera = Camera("ortho", theta=theta, phi=phi, scale=scale)
        mesh = meshio.read(filename)
        vertices = glm.fit_unit_cube(mesh.points)
        faces = mesh.cells[0].data
        vertices = glm.fit_unit_cube(vertices)
        mesh = Mesh(
            ax,
            camera.transform,
            vertices,
            faces,
            cmap=plt.get_cmap(colormap),
            edgecolors=edgecolors,
        )
        plt.show()  # return fig or ax instead?


    show("data/bunny.obj", scale=1.5)
    return (show,)


@app.cell
def __(mo):
    mo.md("""## OBJ Format""")
    return


@app.cell
def __():
    with open('data/bunny.obj', mode='tr') as _file:
        bunny_obj = _file.read()
    _lines = bunny_obj.splitlines()
    for _line in _lines[:10]:
        print(_line)
    print('...')
    for _line in _lines[-10:]:
        print(_line)
    return (bunny_obj,)


@app.cell
def __(np):
    def parse_obj(obj_filename):
        vertices = []
        indices = []
        with open(obj_filename, mode="tr") as file:
            lines = file.read().splitlines()
        for line in lines:
            if line.startswith("#"):
                pass
            elif line.startswith("v"):
                coords = [np.float32(f) for f in line.split()[1:]]
                vertices.append(coords)
            elif line.startswith("f"):
                _indices = [int(i) for i in line.split()[1:]]
                indices.append(_indices)
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices)
        return vertices, indices
    return (parse_obj,)


@app.cell
def __(parse_obj):
    parse_obj("data/bunny.obj")
    return


@app.cell
def __(make_STL, np, parse_obj):
    def OBJ_to_STL(obj_filename, stl_filename):
        vertices, indices = parse_obj(obj_filename)
        faces = np.zeros((len(indices), 3, 3), dtype=np.float32)
        for index, (i, j, k) in enumerate(indices):
            faces[index] = [vertices[i - 1], vertices[j - 1], vertices[k - 1]]
        stl = make_STL(faces)
        with open(stl_filename, "tw") as file:
            file.write(stl)
    return (OBJ_to_STL,)


@app.cell
def __(OBJ_to_STL, show):
    OBJ_to_STL("data/bunny.obj", "output/bunny.stl")
    show("output/bunny.stl", scale=1.5)
    return


@app.cell
def __(meshio):
    mesh = meshio.read("output/bunny.stl") # ⚠️ no blank line in STL for meshio!
    return (mesh,)


@app.cell
def __(mo):
    mo.md(r"""## Constructive Solid Geometry (CSG)""")
    return


@app.cell
def __(X, Y, Z, box, cylinder, sphere):
    demo_csg = sphere(1) & box(1.5)
    _c = cylinder(0.5)
    demo_csg = demo_csg - (_c.orient(X) | _c.orient(Y) | _c.orient(Z))
    demo_csg.save('output/demo-csg.stl', step=0.05)
    return (demo_csg,)


@app.cell
def __(show):
    show("output/demo-csg.stl", theta=45.0, phi=45.0, scale=1.0)
    return


@app.cell
def __(box, cylinder, difference, intersection, orient, sphere, union):
    demo_csg_alt = difference(
        intersection(
            sphere(1),
            box(1.5),
        ),
        union(
            orient(cylinder(0.5), [1.0, 0.0, 0.0]),
            orient(cylinder(0.5), [0.0, 1.0, 0.0]),
            orient(cylinder(0.5), [0.0, 0.0, 1.0]),
        ),
    )
    demo_csg_alt.save("output/demo-csg-alt.stl", step=0.05)
    return (demo_csg_alt,)


@app.cell
def __(show):
    show("output/demo-csg-alt.stl", theta=45.0, phi=45.0, scale=1.0)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Binary STL

        - [ ] Discover that `out.stl` is not ASCII text, but binary
        - [ ] Read about binary STL online
        - [ ] Ask if there is anything useful in the first 80 bytes (header)
        - [ ] Read the number of triangles `n` (hint to `numpy.fromfile`); check that works out.
        - [ ] Check that the lenth of the binary data checks out with this count and the spec
        - [ ] Read the numeric data as a `(n, 12)` array of `float32`
        - [ ] Extract from this array the `normals` (shape `(n, 3)`) and `vertices` (shape `(n, 3, 3)`) arrays.
        - [ ] At the end, measure the compression ratio offered by binary?
        """
    )
    return


@app.cell
def __(show):
    show("data/dragon.stl", theta=75.0, phi=-20.0, scale=1.7)
    return


@app.cell
def __(mo):
    mo.md("""TODO: STL binary to STL text; use it on teapot to get a reference text file (dragon is already in binary). Also ask for the opposite? (text to binary)""")
    return


@app.cell
def __(make_STL, np):
    def STL_binary_to_text(stl_filename_in, stl_filename_out):
        with open(stl_filename_in, mode="rb") as file:
            _ = file.read(80)
            n = np.fromfile(file, dtype=np.uint32, count=1)[0]
            print(n)
            normals = []
            faces = []
            for i in range(n):
                normals.append(np.fromfile(file, dtype=np.float32, count=3))
                faces.append(np.fromfile(file, dtype=np.float32, count=9).reshape(3, 3))
                _ = file.read(2)
        stl_text = make_STL(faces, normals)
        with open(stl_filename_out, mode="wt", encoding="utf-8") as file:
            file.write(stl_text)
    return (STL_binary_to_text,)


@app.cell
def __(STL_binary_to_text, show):
    STL_binary_to_text("data/dragon.stl", "output/dragon.stl")
    show("output/dragon.stl", theta=75.0, phi=-20.0, scale=1.7)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## JCAD

        - [ ] Get the basic (provided) box, sphere, (capped) cylinder jcad models and build a object for each of them.
        - [ ] Automate the conversion with a jcad_to_sdf function that works for each of these primitives. Make sure that the function still works if you change the length/width/height parameters of the box, the radius of the sphere and the radius and
        height of the cylinder.
        - [ ] Improve your function so that it also supports the angle 1, 2 and 3 of the sphere and the angle of the cylinder. (Make it optional or get rid of this, we don't care)
        - [ ] Extend your function so that all the placements parameters are also supported.
        - [ ] Extend your function so that it can work with several primitive objects. You will keep in the sdf object only the (union of) the objects that are visible in the jcad model.
        - [ ] Make your function work with complex models based on union, intersection and differences of primitive objects.

        "Validation":

        - [ ] Reproduce the canonical CSG example with JCAD, save it as "data/csg.jcad"
        - [ ] Output "the" (or the union of) the objects that are visible in the JCAD GUI
        - [ ] ...

        TODO: present this upside down: start with the design of the canonical example in jcad, then work your way up in the conversion.
        """
    )
    return


@app.cell
def __(np, sdf):
    def flat_jcad_to_sdf(jcad):
        shapes = jcad["objects"]
        parts = {}
        for shape in shapes:
            kind = shape["shape"]
            params = shape["parameters"]
            if kind == "Part::Box":
                length = params["Length"]
                width = params["Width"]
                height = params["Height"]
                part = sdf.box((length, width, height)).translate(
                    (0.5 * length, 0.5 * width, 0.5 * height)
                )
            elif kind == "Part::Sphere":
                radius = params["Radius"]
                part = sdf.sphere(radius=radius)
            elif kind == "Part::Cylinder":
                height = params["Height"]
                radius = params["Radius"]
                part = sdf.capped_cylinder([0, 0, 0], [0, 0, height], radius)
            elif kind == "Part::MultiFuse":
                shapes = [parts[name] for name in params["Shapes"]]
                part = sdf.union(*shapes)
            elif kind == "Part::MultiCommon":
                shapes = [parts[name] for name in params["Shapes"]]
                part = sdf.intersection(*shapes)
            elif kind == "Part::Cut":
                base = parts[params["Base"]]
                tool = parts[params["Tool"]]
                part = sdf.difference(base, tool)
            placement = params["Placement"]
            offset = placement["Position"]
            rotation_axis = placement["Axis"]
            rotation_angle = placement["Angle"] / 180.0 * np.pi
            part = part.rotate(rotation_angle, rotation_axis).translate(offset)
            name = shape["name"]
            parts[name] = part
        visibility_map = jcad["options"]["guidata"]
        include_parts = [
            parts[name]
            for name, spec in visibility_map.items()
            if spec["visibility"] == True
        ]
        return sdf.union(*include_parts)
    return (flat_jcad_to_sdf,)


@app.cell
def __(flat_jcad_to_sdf, json):
    with open("data/jcad/csg.jcad", mode="rt", encoding='utf-8') as _file:
        jcad_model = json.load(_file)
    sdf_model = flat_jcad_to_sdf(jcad_model)
    sdf_model.save("output/demo_jcad.stl", step=0.05)
    return jcad_model, sdf_model


@app.cell
def __(show):
    show("output/demo_jcad.stl", theta=45.0, phi=45.0)
    return


@app.cell
def __(mo):
    mo.md("""## Annex""")
    return


@app.cell
def __():
    ### Dependencies
    return


@app.cell
def __():
    # Python Standard Library
    import json

    # Marimo
    import marimo as mo

    # Third-Party Librairies
    import numpy as np; np.seterr(over="ignore")
    import matplotlib.pyplot as plt
    import mpl3d
    from mpl3d import glm
    from mpl3d.mesh import Mesh
    from mpl3d.camera import Camera
    import meshio
    import sdf
    from sdf import sphere, box, cylinder
    from sdf import X, Y, Z
    from sdf import intersection, union, orient, difference
    return (
        Camera,
        Mesh,
        X,
        Y,
        Z,
        box,
        cylinder,
        difference,
        glm,
        intersection,
        json,
        meshio,
        mo,
        mpl3d,
        np,
        orient,
        plt,
        sdf,
        sphere,
        union,
    )


if __name__ == "__main__":
    app.run()
