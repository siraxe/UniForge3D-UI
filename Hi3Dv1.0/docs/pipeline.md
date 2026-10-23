# How the pipeline works

Hi3D v1.0 refines a coarse mesh into high-resolution geometry, conditioned on a
reference image of the same object. It retains the Sparc3D representation and
refinement architecture. The Python package and API use `sparc3d` and
`Sparc3DPipeline`.

A refinement *pass* takes occupied voxel coordinates, samples a latent at each
coordinate, decodes the latents into a signed-distance field, and extracts a
mesh. Revoxelizing that mesh provides the sparse support for the next pass.

| Pass | Nominal structure grid | Voxel budget | Transformer | Nominal extraction grid |
| --- | ---: | ---: | --- | ---: |
| 1 | 64³ | 80,000 | `transformer_512` | 512³ |
| 2 | 128³ | 90,000 | `transformer_1024` | 1024³ |
| 3 | 192³ | 110,000 | `transformer_1024` | 1536³ |

Requesting 512³ runs one pass, 1024³ runs two, and 1536³ runs all three. Each pass
uses the preceding mesh to place its sparse coordinates. The `transformer_1024`
model serves both the 128³ and 192³ structural grids; its configuration names
128, while its coordinate-based architecture also supports the third pass.
These grid labels do not indicate a dense latent tensor or a fixed output
triangle count.

## Stage 1: coarse mesh

The reference image and coarse mesh must describe the same object. The mesh
supplies global structure; refinement may retain severe structural errors in
that initialization. Unseen surfaces are plausible completions rather than
exact reconstructions of hidden geometry.

Coarse generation is external to the library. The optional
`demo/coarse_stage.py` adapter uses Hunyuan3D-2 mini, and
`demo/image_to_mesh.py` invokes it before refinement. Hunyuan is neither a core
dependency nor a required product integration. The adapter can be replaced;
`demo/refine_mesh.py` and `Sparc3DPipeline.__call__` accept an existing triangle
mesh directly without Hunyuan.

Hunyuan code and weights are obtained separately and retain their upstream
licenses; they are not released or relicensed by Hi3D. See the
[coarse-stage setup](installation.md#the-coarse-stage) and
[external-model terms](../THIRD_PARTY_LICENSES.md#external-models-and-optional-examples).

## Stage 2: conditioning

`prepare_image` uses the alpha bounding box to crop the subject to a square,
composites it over white, and pads the square so the subject occupies
`1 / 1.2` of the side length. This framing matches the model's conditioning
preprocessing and affects the resulting image tokens.

An image without alpha is treated as fully opaque, so its entire frame,
including its background, is used. Background removal is optional and must be
requested with `--remove-background` in the demos after installing the `[demo]`
extra.

`ImageConditioner` runs two frozen encoders in fp32 with enclosing autocast
disabled:

- **DINOv2-large:** 756×756 input, producing 2,917 image tokens of 1,024 channels.
- **CLIP ViT-L/14:** 224×224 input, producing a pooled 1,024-channel embedding
  broadcast across the 2,917 positions.

The features are L2-normalized and concatenated into a `[1, 2917, 2048]` tensor,
computed once and reused by all passes. These encoders are separate model
artifacts; see [encoder loading and offline setup](installation.md#the-conditioning-encoders).

## Stage 3: sparse flow sampling

Each occupied voxel receives a 32-channel latent sampled by a sparse
flow-matching transformer. The two transformer models use 24 blocks, a width of
2,112 channels, and 22 attention heads, with their main layers in fp16 and
approximately 1.8 billion parameters each.

The checkpoint architecture uses rotary position embeddings in self-attention,
flash attention, RMS normalization of self-attention queries and keys, shared
timestep modulation, and skip connections between the input and output
resolution blocks. Each block applies sparse self-attention over voxels,
cross-attention to image tokens, and an MLP.

Sampling defaults to **25 scheduler updates per pass** with the second-order
`FlowMatchHeunScheduler`. Each update computes an Euler predictor and averages
the flow at the current and predicted states. The first-order
`FlowMatchEulerScheduler` can be selected in
`scheduler/scheduler_config.json`; both use the same sigma schedule.

Classifier-free guidance evaluates an additional zero-condition branch only
when `sigma >= 0.7`. Each Heun slope applies this threshold at its own sigma.
After sampling, per-channel means and standard deviations from the transformer
configuration map the latents back to the VAE latent space.

## Stage 4: VAE decode and mesh extraction

The inference package includes the `SparconvVAE` decoder.
`PTv3MeshDecoder` progressively upsamples sparse 32-channel latents. At each
upsampling stage it predicts occupancy for eight child voxels and prunes empty
regions. Three 2× stages map a nominal 64³ structural grid to a 512³ extraction
grid without materializing a dense 512³ volume.

Decoder attention uses fixed-size windows over a space-filling-curve ordering
of the sparse voxels. The decoder predicts per-corner signed distances and
vertex deformations. FlexiCubes extracts a triangle mesh, and ear clipping fills
remaining holes. Between passes, direct surface voxelization of the extracted
mesh produces the next sparse support.

## Stage 5: postprocessing

The final mesh is filtered, simplified, and normalized:

1. Disconnected components with fewer than 0.5% of all faces are removed; the
   largest component is always retained.
2. `pyfqmr` simplifies toward 1,000,000 faces at 512³ or 1024³, and 2,000,000 at
   1536³. These are simplification targets, not guaranteed exact counts.
3. The mean vertex position is subtracted, then vertices are scaled so the
   maximum radius is one. Input world-space position and absolute scale are
   not preserved.

Output contains geometry only, without materials, textures, or vertex colors.

## Progress reporting

Pass a `progress_callback` to `Sparc3DPipeline` or `from_pretrained` to receive a
frozen `Sparc3DProgressEvent` after each scheduler update. The event includes
`pass_index`, `num_passes`, `structure_resolution`, one-based `step`, and
`num_inference_steps`.

Progress counts scheduler updates rather than model evaluations: a 25-step Heun
pass reports 25 events. Decoding, extraction, and postprocessing do not emit
these events, so progress displays may pause between sampling passes.

## Memory

The pipeline uses batch size one. Peak GPU memory depends on both model
residency and decoded sparse feature counts. Before proceeding to another
pass, it raises if the decoded feature count exceeds 12,000,000 rather than
returning an output below the requested resolution. This guard does not
replace checking available GPU memory.

High-resolution transformer loading is deferred until needed in the standard
pipeline. With Hub artifacts, this also defers its download. A local artifact
directory must contain all required files.
`enable_sequential_cpu_offload()` materializes both transformers on the CPU and
installs Accelerate hooks to move components to the GPU as needed. Enable it
before moving models to a device; it cannot be combined with `.to("cuda")`.
Offload reduces model residency but not all decoded-feature memory costs.

The 1536³ ladder has been validated only on a **96 GB NVIDIA H20**. Peak memory
and feasibility on smaller cards have not been established; see the
[validated environment](installation.md#the-verified-stack).

Voxelization is adaptive. If occupied coordinates exceed a pass's voxel budget,
the grid resolution is reduced by 5% and voxelization retries. Complex geometry
can therefore use coarser support than the nominal structural grid, with a
warning. The requested extraction-resolution label remains unchanged.

See [reproduction and validation](reproduction.md) for GPU nondeterminism,
run-to-run comparison, and the inference-only evaluation scope.

Adapted from SPARC3D documentation for Hi3D v1.0.
