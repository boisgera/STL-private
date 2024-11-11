import marimo

__generated_with = "0.9.11"
app = marimo.App()


@app.cell
def __():
    # STL Project
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## About STL

        STL is a simple file format which describes 3D objects as a collection of triangles.
        The acronym STL stands for "Simple Triangle Language", "Standard Tesselation Language" or "STereoLitography"[^1].

        [^1]: STL was invented for – and is still widely used – for 3D printing.
        """
    )
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/teapot.stl", theta=45.0, phi=30.0, scale=2))
    return


@app.cell
def __(mo):
    with open("data/teapot.stl", mode="rt", encoding="utf-8") as _file:
        teapot_stl = _file.read()

    teapot_stl_excerpt = teapot_stl[:723] + "..." + teapot_stl[-379:]

    mo.md(
        f"""
    ## STL ASCII Format

    The `data/teapot.stl` file provides an example of the STL ASCII format. It is quite large (more than 60000 lines) and looks like that:
    """
    +
    f"""```
    {teapot_stl_excerpt}
    ```
    """
    +

    """
    """
    )
    return teapot_stl, teapot_stl_excerpt


@app.cell
def __(mo):
    mo.md(f"""

      - Study the [{mo.icon("mdi:wikipedia")} STL (file format)](https://en.wikipedia.org/wiki/STL_(file_format)) page (or other online references) to become familiar the format.

      - Create a STL ASCII file `"data/cube.stl"` that represents a cube of unit length  
        (💡 in the simplest version, you will need 12 different facets).

      - Display the result with the function `show` (make sure to check different angles).
    """)
    return


@app.cell
def __(mo):
    mo.md(r"""## STL & NumPy""")
    return


@app.cell
def __(mo):
    mo.md(rf"""

    ### NumPy to STL

    Implement the following function:

    ```python
    def make_STL(triangles, normals=None, name=""):
        pass # 🚧 TODO!
    ```

    #### Parameters

      - `triangles` is a NumPy array of shape `(n, 3, 3)` and data type `np.float32`,
         which represents a sequence of `n` triangles (`triangles[i, j, k]` represents 
         is the `k`th coordinate of the `j`th point of the `i`th triangle)

      - `normals` is a NumPy array of shape `(n, 3)` and data type `np.float32`;
         `normals[i]` represents the outer unit normal to the `i`th facet.
         If `normals` is not specified, it should be computed from `triangles` using the 
         [{mo.icon("mdi:wikipedia")} right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule).

      - `name` is the (optional) solid name embedded in the STL ASCII file.

    #### Returns

      - The STL ASCII description of the solid as a string.

    #### Example

    Given the two triangles that make up a flat square:

    ```python

    square_triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    ```

    then printing `make_STL(square_triangles, name="square")` yields
    ```
    solid square
      facet normal 0.0 0.0 1.0
        outer loop
          vertex 0.0 0.0 0.0
          vertex 1.0 0.0 0.0
          vertex 0.0 1.0 0.0
        endloop
      endfacet
      facet normal 0.0 0.0 1.0
        outer loop
          vertex 1.0 1.0 0.0
          vertex 0.0 1.0 0.0
          vertex 1.0 0.0 0.0
        endloop
      endfacet
    endsolid square
    ```

    """)
    return


@app.cell
def __():
    return


@app.cell
def __(np):
    def make_STL(triangles, normals=None, name=None):
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
            parts.append(
    f"""  facet normal {n[0]} {n[1]} {n[2]}
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
def __(mo):
    mo.md(
        """
        ### STL to NumPy

        Implement a `tokenize` function


        ```python
        def tokenize(stl):
            pass # 🚧 TODO!
        ```

        that is consistent with the following documentation:


        #### Parameters

          - `stl`: a Python string that represents a STL ASCII model.

        #### Returns

          - `tokens`: a list of STL keywords (`solid`, `facet`, etc.) and `np.float32` numbers.

        #### Example

        For the ASCII representation the square `data/square.stl`, printing the tokens with

        ```python
        with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
            square_stl = square_file.read()
        tokens = tokenize(square_stl)
        print(tokens)
        ```

        yields

        ```python
        ['solid', 'square', 'facet', 'normal', np.float32(0.0), np.float32(0.0), np.float32(1.0), 'outer', 'loop', 'vertex', np.float32(0.0), np.float32(0.0), np.float32(0.0), 'vertex', np.float32(1.0), np.float32(0.0), np.float32(0.0), 'vertex', np.float32(0.0), np.float32(1.0), np.float32(0.0), 'endloop', 'endfacet', 'facet', 'normal', np.float32(0.0), np.float32(0.0), np.float32(1.0), 'outer', 'loop', 'vertex', np.float32(1.0), np.float32(1.0), np.float32(0.0), 'vertex', np.float32(0.0), np.float32(1.0), np.float32(0.0), 'vertex', np.float32(1.0), np.float32(0.0), np.float32(0.0), 'endloop', 'endfacet', 'endsolid', 'square']
        ```
        """
    )
    return


@app.cell
def __(np):
    STL_KEYWORDS = [
        "solid",
        "endsolid",
        "facet",
        "endfacet",
        "outer",
        "loop",
        "endloop",
        "vertex",
    ]


    def tokenize(STL_text):
        raw_tokens = STL_text.split()
        tokens = []
        for token in raw_tokens:
            try:  # is the token a number?
                tokens.append(np.float32(token))
            except ValueError:
                tokens.append(token)
        return tokens
    return STL_KEYWORDS, tokenize


@app.cell
def __(mo):
    mo.md(
        """
        Implement a `parse` function


        ```python
        def parse(tokens):
            pass # 🚧 TODO!
        ```

        that is consistent with the following documentation:


        #### Parameters

          - `tokens`: a list of tokens

        #### Returns

        A `triangles, normals, name` triple where

          - `triangles`: a `(n, 3, 3)` NumPy array with data type `np.float32`,

          - `normals`: a `(n, 3)` NumPy array with data type `np.float32`,

          - `name`: a Python string.

        #### Example

        For the ASCII representation `square_stl` of the square,
        tokenizing then parsing

        ```python
        with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
            square_stl = square_file.read()
        tokens = tokenize(square_stl)
        triangles, normals, name = parse(tokens)
        print(repr(triangles))
        print(repr(normals))
        print(repr(name))
        ```

        yields

        ```python
        array([[[0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.]],

               [[1., 1., 0.],
                [0., 1., 0.],
                [1., 0., 0.]]], dtype=float32)
        array([[0., 0., 1.],
               [0., 0., 1.]], dtype=float32)
        'square'
        ```
        """
    )
    return


@app.cell
def __(np):
    def parse(tokens):
        name = ""
        if tokens[2] == "facet":
            name = tokens[1]
            del tokens[1]
        normals = []
        for i, token in enumerate(tokens):
            if token == "normal":
                normals.append(np.array(tokens[i+1:i+4]))
        triangles = []
        for i, token in enumerate(tokens):
            if token == "loop":
                triangle = []
                triangle.append(np.array(tokens[i+2:i+5]))
                triangle.append(np.array(tokens[i+6:i+9]))
                triangle.append(np.array(tokens[i+10:i+13]))
                triangles.append(np.array(triangle))
        return np.array(triangles), np.array(normals), name
    return (parse,)


@app.cell
def __(parse, tokenize):
    def from_STL(filename):
        with open(filename, mode="tr", encoding="us_ascii") as file:
            text = file.read()
        tokens = tokenize(text)
        return parse(tokens)
    return (from_STL,)


@app.cell
def __(mo):
    mo.md(
        rf"""
    ## Rules & Diagnostics



        Make diagnostic functions that check whether a STL model satisfies the following rules

          - **Positive octant rule.** All vertex coordinates are non-negative.

          - **Orientation rule.** All normals are (approximately) unit vectors and follow the [{mo.icon("mdi:wikipedia")} right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule).
        
          - **Shared edge rule.** Each triangle edge appears exactly twice.

          - **Ascending rule.** the z-coordinates of (the barycenter of) each triangle are a non-decreasing sequence.

    When the rule is broken, make sure to display some sensible quantitative measure of the violation (in %).

    For the record, the `data/teapot.STL` file:

      - does not obey the positive octant rule,
      - almost obeys the orientation rule, 
      - obeys the shared edge rule,
      - does not obey the ascending rule.

    Check that your `data/cube.stl` file does follow all these rules, or modify it accordingly!

    """
    )
    return


@app.cell
def __(from_STL, np):
    def check_orientation(filename, verbose=True, tolerance=1e-4):
        triangles, normals, _ = from_STL(filename)
        d1 = triangles[:, 1, :] - triangles[:, 0, :]
        d2 = triangles[:, 2, :] - triangles[:, 1, :]
        vector_product = np.linalg.cross(d1, d2)
        norms = np.linalg.norm(vector_product, axis=1)
        computed_normals = np.diag(1 / norms) @ vector_product
        e = np.linalg.vector_norm(normals - computed_normals, axis=1)
        status = all(e <= tolerance)
        if verbose:
            if status:
                print("🟢 0% error")
            else:
                error = (e > tolerance).sum() / len(e) * 100.0
                print(f"🔴 {error:.2f}% error")
        return status
    return (check_orientation,)


@app.cell
def __(check_orientation):
    check_orientation("data/teapot.stl")
    return


@app.cell
def __(from_STL):
    def check_shared_edges(filename, verbose=True):
        triangles, _, _ = from_STL(filename)
        count = {}
        for t in triangles:
            edges = [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]]
            for edge in edges:
                p1, p2 = edge
                p1 = tuple([float(x) for x in p1])
                p2 = tuple([float(x) for x in p2])
                edge = tuple(sorted((p1, p2)))
                count[edge] = count.get(edge, 0) + 1
        status = set(count.values()) == {2}
        if verbose:
            if status:
                print("🟢 0% error")
            else:
                error = sum([c for c in count.values() if c != 2]) / sum(count.values()) * 100.0
                print(f"🔴 {error:.2f}% error")
        return status
    return (check_shared_edges,)


@app.cell
def __(check_shared_edges):
    check_shared_edges("data/teapot.stl")
    return


@app.cell
def __(from_STL, np):
    def check_octant(filename, verbose=True):
        triangles, normals, _ = from_STL(filename)
        coords = np.reshape(triangles, (-1,))
        ok = all(coords>=0)
        if verbose:
            if ok:
                print("🟢 0% error")
            else:
                error = (coords<0).sum() / len(coords) * 100.0  
                print(f"🔴 {error:.2f}% error")
        return ok
    return (check_octant,)


@app.cell
def __(check_octant):
    check_octant("data/teapot.stl")
    return


@app.cell
def __(from_STL, np):
    def check_ascending(filename, verbose=True):
        triangles, normals, _ = from_STL(filename)
        z = triangles.mean(axis=1)[:,-1]
        d = np.diff(z)
        error_count = (d < 0).sum()
        if verbose:
            if error_count:
                error = error_count / len(d) * 100.0  
                print(f"🔴 {error:.2f}% error")
            else:
                print("🟢 0% error")
        return not error_count
    return (check_ascending,)


@app.cell
def __(check_ascending):
    check_ascending("data/teapot.stl")
    return


@app.cell
def __(check_orientation):
    check_orientation("data/square.stl")
    return


@app.cell
def __(check_shared_edges):
    check_shared_edges("data/square.stl")
    return


@app.cell
def __(check_octant):
    check_octant("data/square.stl")
    return


@app.cell
def __(check_ascending):
    check_ascending("data/square.stl")
    return


@app.cell
def __(
    check_ascending,
    check_octant,
    check_orientation,
    check_shared_edges,
):
    _cube = "data/cube.stl"
    check_orientation(_cube)
    check_shared_edges(_cube)
    check_octant(_cube)
    check_ascending(_cube)
    return


@app.cell
def __(mo):
    mo.md(
    rf"""
    ## OBJ Format

    The OBJ format is an alternative to the STL format that looks like this:

    ```
    # OBJ file format with ext .obj
    # vertex count = 2503
    # face count = 4968
    v -3.4101800e-003 1.3031957e-001 2.1754370e-002
    v -8.1719160e-002 1.5250145e-001 2.9656090e-002
    v -3.0543480e-002 1.2477885e-001 1.0983400e-003
    v -2.4901590e-002 1.1211138e-001 3.7560240e-002
    v -1.8405680e-002 1.7843055e-001 -2.4219580e-002
    ...
    f 2187 2188 2194
    f 2308 2315 2300
    f 2407 2375 2362
    f 2443 2420 2503
    f 2420 2411 2503
    ```

    This content is an excerpt from the `data/bunny.obj` file.

    """
    )
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/bunny.obj", scale="1.5"))
    return


@app.cell
def __(mo):
    mo.md(
        """
        Study the specification of the OBJ format (search for suitable sources online),
        then develop a `OBJ_to_STL` function that is rich enough to convert the OBJ bunny file into a STL bunny file.
        """
    )
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
def __(mo):
    mo.md(
        rf"""
    ## Binary STL

    Since the STL ASCII format can lead to very large files when there is a large number of facets, there is an alternate, binary version of the STL format which is more compact.

    Read about this variant online, then implement the function

    ```python
    def STL_binary_to_text(stl_filename_in, stl_filename_out):
        pass  # 🚧 TODO!
    ```

    that will convert a binary STL file to a ASCII STL file. Make sure that your function works with the binary `data/dragon.stl` file which is an example of STL binary format.

    💡 The `np.fromfile` function may come in handy.

        """
    )
    return


@app.cell
def __(mo, show):
    mo.show_code(show("data/dragon.stl", theta=75.0, phi=-20.0, scale=1.7))
    return


@app.cell
def __(make_STL, np):
    def STL_binary_to_text(stl_filename_in, stl_filename_out):
        with open(stl_filename_in, mode="rb") as file:
            _ = file.read(80)
            n = np.fromfile(file, dtype=np.uint32, count=1)[0]
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
def __(STL_binary_to_text, mo, show):
    STL_binary_to_text("data/dragon.stl", "output/dragon.stl")
    mo.show_code(show("output/dragon.stl", theta=75.0, phi=-20.0, scale=1.7))
    return


@app.cell
def __(mo):
    mo.md(rf"""## Constructive Solid Geometry (CSG)

    Have a look at the documentation of [{mo.icon("mdi:github")}fogleman/sdf](https://github.com/fogleman/) and study the basics. At the very least, make sure that you understand what the code below does:
    """)

    return


@app.cell
def __(X, Y, Z, box, cylinder, mo, show, sphere):
    demo_csg = sphere(1) & box(1.5)
    _c = cylinder(0.5)
    demo_csg = demo_csg - (_c.orient(X) | _c.orient(Y) | _c.orient(Z))
    demo_csg.save('output/demo-csg.stl', step=0.05)
    mo.show_code(show("output/demo-csg.stl", theta=45.0, phi=45.0, scale=1.0))
    return (demo_csg,)


@app.cell
def __(mo):
    mo.md("""ℹ️ **Remark.** The same result can be achieved in a more procedural style, with:""")
    return


@app.cell
def __(
    box,
    cylinder,
    difference,
    intersection,
    mo,
    orient,
    show,
    sphere,
    union,
):
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
    mo.show_code(show("output/demo-csg-alt.stl", theta=45.0, phi=45.0, scale=1.0))
    return (demo_csg_alt,)


@app.cell
def __(mo):
    mo.md(
        rf"""
    ## JupyterCAD

    [JupyterCAD](https://github.com/jupytercad/JupyterCAD) is an extension of the Jupyter lab for 3D geometry modeling.

      - Use it to create a JCAD model that correspond closely to the `output/demo_csg` model;
    save it as `data/demo_jcad.jcad`.

      - Study the format used to represent JupyterCAD files (💡 you can explore the contents of the previous file, but you may need to create some simpler models to begin with).

      - When you are ready, create a `jcad_to_stl` function that understand enough of the JupyterCAD format to convert `"data/demo_jcad.jcad"` into some corresponding STL file.
    (💡 do not tesselate the JupyterCAD model by yourself, instead use the `sdf` library!)


        """
    )
    return


@app.cell
def __(np, sdf):
    def jcad_to_sdf(jcad):
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
    return (jcad_to_sdf,)


@app.cell
def __(jcad_to_sdf, json):
    with open("data/demo-jcad.jcad", mode="rt", encoding='utf-8') as _file:
        jcad_model = json.load(_file)
    sdf_model = jcad_to_sdf(jcad_model)
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
def __(mo):
    mo.md("""### Dependencies""")
    return


@app.cell
def __():
    # Python Standard Library
    import json

    # Marimo
    import marimo as mo

    # Third-Party Librairies
    import numpy as np


    import matplotlib.pyplot as plt
    import mpl3d
    from mpl3d import glm
    from mpl3d.mesh import Mesh
    from mpl3d.camera import Camera

    import meshio

    np.seterr(over="ignore")  # 🩹 deal with a meshio false warning

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


@app.cell
def __(Camera, Mesh, glm, meshio, mo, plt):
    def show(
        filename,
        theta=0.0,
        phi=0.0,
        scale=1.0,
        colormap="viridis",
        edgecolors=(0, 0, 0, 0.25),
        figsize=(6, 6),
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
        return mo.center(fig)
    return (show,)


@app.cell
def __(mo):
    mo.md(r"""### STL Viewer""")
    return


@app.cell
def __(show):
    show("data/teapot.stl", theta=45.0, phi=30.0, scale=2)
    return


@app.cell
def __(mo):
    mo.md("""### Sandbox""")
    return


@app.cell
def __(make_STL, np):
    square_triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    square_stl = make_STL(square_triangles, name="square")
    print(square_stl)
    with open("data/square.stl", mode="wt", encoding="us-ascii") as _file:
        _file.write(square_stl)
    return square_stl, square_triangles


@app.cell
def __(make_STL, np):
    _h = np.sqrt(2)/2

    pyramid_triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, _h]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.5, 0.5, _h]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, _h]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, _h]],
        ],
        dtype=np.float32,
    )

    pyramid = "data/pyramid.stl"

    with open(pyramid, mode="tw", encoding="utf-8") as _file:
        _file.write(make_STL(pyramid_triangles, name="pyramid"))
    return pyramid, pyramid_triangles


@app.cell
def __(pyramid, show):
    show(pyramid, theta=60.0, phi=30.0, scale=1.2)
    return


@app.cell
def __(pyramid):
    open(pyramid, mode="rt", encoding="us-ascii").read()
    return


@app.cell
def __(make_STL, np):
    cube_triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
            [[0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )

    cube_normals = np.array(
        [
            [+0.0, +0.0, -1.0],
            [+0.0, +0.0, -1.0],
            [+0.0, -1.0, +0.0],
            [+1.0, +0.0, +0.0],
            [+0.0, +1.0, +0.0],
            [-1.0, +0.0, +0.0],
            [+0.0, -1.0, +0.0],
            [+1.0, +0.0, +0.0],
            [+0.0, +1.0, +0.0],
            [-1.0, +0.0, +0.0],
            [+0.0, +0.0, +1.0],
            [+0.0, +0.0, +1.0],
        ],
        dtype=np.float32,
    )

    cube = "output/cube.stl"

    with open(cube, mode="tw", encoding="utf-8") as _file:
        _file.write(make_STL(cube_triangles, cube_normals, name="cube"))
    return cube, cube_normals, cube_triangles


@app.cell
def __(cube, show):
    show(cube, phi=30.0, theta=60.0, scale=1.0)
    return


@app.cell
def __(check_orientation, cube):
    check_orientation(cube)
    return


@app.cell
def __(check_shared_edges, cube):
    check_shared_edges(cube)
    return


@app.cell
def __(check_octant, cube):
    check_octant(cube)
    return


@app.cell
def __(check_ascending, cube):
    check_ascending(cube)
    return


if __name__ == "__main__":
    app.run()
