# Inference validation and reproducibility

Hi3D v1.0 supports inference validation with a reference image, a coarse mesh,
compatible model artifacts, and the environment described in
[installation](installation.md). Model weights must be obtained separately.

Record the following when comparing runs:

- Input image and coarse mesh, including any background removal or preprocessing.
- Refinement model version, image-encoder versions, and source-code version.
- Output resolution, scheduler, inference-step count, and random seed.
- Python, PyTorch, CUDA, GPU, and dependency versions.
- For end-to-end generation, the coarse generator, its model version, and its settings.

## Reference environment

The following environment was used for inference validation:

| Component | Version |
| --- | --- |
| Python | 3.10.12 |
| torch | 2.5.0+cu124 |
| torchvision | 0.20.0+cu124 |
| CUDA toolkit | 12.4.131 |
| spconv-cu120 | 2.3.6 |
| flash-attn | 2.7.4.post1 |
| transformers | 4.46.3 |
| numpy | 1.26.4 |
| GPU | NVIDIA H20 (`sm_90`) |

`spconv-cu120` and `flash-attn` are pinned in the `inference` extra. Use
`docker/constraints.txt` when installing the reference dependency versions; see
[installation](installation.md) for other CUDA and kernel configurations.

## Random seeds and GPU variation

The refinement sampler draws its initial noise from a local CPU
`torch.Generator`. Fixing `seed` reproduces that sampling initialization without
changing the global PyTorch RNG state.

Sparse convolution and flash attention on GPU can still produce numerical
differences between runs. The same seed and environment therefore do not
guarantee identical meshes or file hashes. Measure run-to-run differences for the
inputs and configuration you use, and compare geometry rather than file hashes.

## Run the same refinement twice

Use the same image, coarse mesh, model directory, resolution, scheduler, step
count, and seed for both runs. For example:

```bash
python demo/refine_mesh.py \
    --model /path/to/hi3d-v1.0-model \
    --image input.png \
    --coarse-mesh coarse.glb \
    --output refined-a.glb \
    --resolution 1024 \
    --num-inference-steps 25 \
    --seed 0
```

Repeat the command with `--output refined-b.glb`. Replace the model path with a
compatible local artifact directory. To validate another supported resolution,
repeat both runs at that resolution while keeping the other settings fixed.

Check that each result reports the requested resolution, has finite vertices and
valid triangular face indices, and can be exported and reloaded. The resolution
is the nominal sparse extraction-grid level, not a triangle count.

## Compare exported meshes

The example below validates the exported mesh structure and compares symmetric
nearest-vertex distances. It uses NumPy, trimesh, and SciPy; install SciPy
separately for this comparison if needed.

Each mesh is centered on its vertex mean and scaled to unit maximum vertex
radius, matching the refinement output convention. This compares normalized
shape, not world-space position or absolute scale. Nearest-vertex distances
depend on vertex sampling and triangulation; they are not the paper's
surface-sampling evaluation metrics.

```python
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def normalized_vertices(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"{path}: expected nonempty 3D vertices")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{path}: nonfinite vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError(f"{path}: expected nonempty triangular faces")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f"{path}: noninteger face indices")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"{path}: face index out of bounds")

    vertices = vertices - vertices.mean(axis=0)
    radius = np.linalg.norm(vertices, axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"{path}: invalid mesh extent")
    return vertices / radius


def symmetric_vertex_distances(path_a, path_b):
    a = normalized_vertices(path_a)
    b = normalized_vertices(path_b)
    a_to_b = cKDTree(b).query(a, k=1)[0]
    b_to_a = cKDTree(a).query(b, k=1)[0]
    return np.concatenate([a_to_b, b_to_a])


d = symmetric_vertex_distances("refined-a.glb", "refined-b.glb")
print(f"mean={d.mean():.6f} p95={np.percentile(d, 95):.6f} max={d.max():.6f}")
```

Report the mean, p95, and maximum distance together with the run configuration.
These describe the selected pair of runs; this release does not prescribe a
universal quality or reproducibility threshold. Inspect the meshes from several
views as well as comparing the numerical distances.

## Optional end-to-end generation

`demo/image_to_mesh.py` first generates a coarse mesh and then runs Hi3D
refinement. Its optional adapter uses Hunyuan3D-2 mini; Hunyuan's implementation
and weights must be obtained separately under their own terms. It is not a core
Hi3D dependency. See the [coarse-stage installation notes](installation.md#the-coarse-stage)
and [external model license index](../THIRD_PARTY_LICENSES.md#external-models-and-optional-examples).

```bash
python demo/image_to_mesh.py \
    --model /path/to/hi3d-v1.0-model \
    --image assets/example.png \
    --coarse-output coarse.glb \
    --output refined.glb \
    --resolution 1024 \
    --num-inference-steps 25 \
    --seed 0
```

Save the intermediate coarse mesh when comparing end-to-end results. Changing
the coarse generator or its output changes the input to the refiner and can
change the final mesh. To compare the refinement stage alone, reuse one saved
coarse mesh with `demo/refine_mesh.py`.

The image encoders are loaded separately from the refinement artifact. Offline
refinement requires `--local-files-only` and local directories supplied through
`--dino-model` and `--clip-model`. Offline end-to-end generation also requires
a local coarse model supplied through `--coarse-model` and the appropriate
`--coarse-subfolder`; the refinement loader flags do not configure Hunyuan's
loader.

## Evaluation scope

This release supports inference validation. It does not include the paper's
Chamfer-distance, absolute-normal-consistency, or F-score evaluation harness,
VAE evaluation protocol, ABO/Objaverse object-ID lists, or comparison meshes.
Reproducing the comparative tables requires those additional resources and a
matched evaluation setup. The structural and run-to-run checks above do not
replace that evaluation.

Adapted from SPARC3D documentation for Hi3D v1.0.
