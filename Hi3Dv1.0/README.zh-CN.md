# Hi3D v1.0

<a href="README.md"><kbd>English</kbd></a> · **简体中文**

**基于 Sparc3D 的高分辨率 3D 几何细化模型。**

**[在线体验](https://huggingface.co/spaces/ilcve21/Sparc3D)** · [Hugging Face 项目主页](https://huggingface.co/hi3d-official/Hi3Dv1.0) · [国际官网](http://hi3d.ai/) · [国内官网](https://www.hi3d.cn/)

已于 2026 年 10 月 23 日公开发布。

## 项目介绍

Hi3D v1.0 以同一物体的参考图和粗网格为输入，细化生成高分辨率几何。模型基于 Sparc3D 的表示与细化架构，逐级采样稀疏潜变量，通过自剪枝稀疏 VAE 解码并提取网格。

项目报告采用约 **200 万个精选 3D 资产**的训练规模，并支持最高 **1536³** 有效稀疏提取网格分辨率。为保持兼容，Python 包与公开 API 继续使用 `sparc3d` 和 `Sparc3DPipeline` 名称。

本仓库提供**推理代码**，不包含训练代码和论文评测工具。模型文件已发布至 [Hugging Face](https://huggingface.co/hi3d-official/Hi3Dv1.0)。示例直接加载 `hi3d-official/Hi3Dv1.0`，也支持传入兼容的本地模型目录。

## 系统框架

[![Hi3D v1.0 系统框架：参考图与粗网格输入、稀疏细化与三个网格提取层级](assets/framework.png)](assets/framework.png)

图像特征在粗几何所确定的稀疏坐标上为流模型提供条件。解码与网格提取后，重新体素化网格，形成下一轮的稀疏支撑。

| 轮次 | 标称结构网格 | 稀疏提取网格 | Transformer |
| --- | --- | --- | --- |
| 1 | 64³ | 512³ | `transformer_512` |
| 2 | 128³ | 1024³ | `transformer_1024` |
| 3 | 192³ | 1536³ | `transformer_1024` |

请求 512³ 时运行一轮，1024³ 运行两轮，1536³ 运行三轮。以上数值表示标称稀疏坐标与提取网格，并非稠密 1536³ 潜变量张量，也不表示固定的三角面数。详见[流程说明](docs/pipeline.md)。

## 效果对比

### 局部几何

[![九种方法的局部几何对比，Hi3D v1.0 以青绿色标出](assets/local-detail-comparison.jpg)](assets/local-detail-comparison.jpg)

红框标出各行放大的参考区域。灰模与彩色表面法线渲染共同展示层叠浮雕、建筑线脚、曲线边界、鳞片、牙齿与同心凹槽等局部结构。**最右列以青绿色标出 Hi3D v1.0。** 点击图片可查看完整分辨率。

### 多视角对比

[![车辆、装甲角色、扶手椅与手提包的多视角几何对比](assets/global-comparison-1.jpg)](assets/global-comparison-1.jpg)

四组物体分别展示正面、背面与底部视角。结果组合呈现灰模与彩色表面法线渲染，彩色部分不表示生成的纹理。隐藏视角展示合理补全，不能视为真实不可见表面的精确重建。

<details>
<summary>更多多视角示例</summary>

[![人形半身像、机械机器人与风格化角色的更多多视角示例](assets/global-comparison-2.jpg)](assets/global-comparison-2.jpg)

更多示例沿用上一张图的列顺序与渲染方式，Hi3D v1.0 仍为青绿色标出的最右列。

</details>

## 安装

环境要求：**Linux、Python 3.10、计算能力不低于 8.0 的 NVIDIA GPU，以及与所装 PyTorch 构建匹配的 CUDA 12.0–12.5 工具链**。不支持 CPU 推理。已实测组合为 Python 3.10.12、PyTorch 2.5.0+cu124、CUDA toolkit 12.4.131 和 **96 GB NVIDIA H20**。1536³ 完整流程仅在该 GPU 上验证，尚无更小显存卡的峰值内存或可用性结论。

克隆本仓库后，在仓库根目录、Python 3.10 环境中执行：

```bash
git clone https://github.com/hi3d-official/Hi3Dv1.0.git
cd Hi3Dv1.0

pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 \
  --extra-index-url https://download.pytorch.org/whl/cu124
pip install packaging ninja psutil
pip install --no-build-isolation flash-attn==2.7.4.post1
pip install -c docker/constraints.txt -e ".[inference]"
```

约束文件保留已实测的内核组合，包括 `spconv-cu120==2.3.6` 与 `flash-attn==2.7.4.post1`。先安装 PyTorch，再单独安装 `flash-attn`：其源码构建需要已安装的 torch 和 CUDA toolkit，并需关闭构建隔离。Triton 网格内核在首次调用时编译。本项目从源码安装，未提供 PyPI 发行包或预编译项目 wheel。

需要演示脚本去背景功能时，另执行 `pip install -c docker/constraints.txt -e ".[demo]"`。该依赖组安装 `rembg`，不安装粗网格生成器。[安装文档](docs/installation.md)提供 Docker、完整验证环境、离线配置与替代 CUDA/内核构建说明；其他配置尚未实测。

## 模型文件

[Hugging Face 模型仓库](https://huggingface.co/hi3d-official/Hi3Dv1.0/tree/main)根目录提供以下 8 个运行文件；下载到本地时请保持相同结构：

```text
model_index.json
scheduler/scheduler_config.json
vae/config.json
vae/model.safetensors
transformer_512/config.json
transformer_512/model.safetensors
transformer_1024/config.json
transformer_1024/model.safetensors
```

加载器读取 JSON 和 safetensors，不执行模型仓库中的 Python，也不反序列化 pickle checkpoint。也可传入具有相同文件结构的 Hugging Face 模型仓库 ID。普通 Hub 加载会延迟加载高分辨率 Transformer；本地目录必须具备全部文件，启用顺序 CPU 卸载时会在 CPU 上实例化两个 Transformer。

条件编码器 `facebook/dinov2-large` 和 `openai/clip-vit-large-patch14` 在首次使用时分别从 Hugging Face 加载，不包含在上述模型目录中。`--revision` 指定细化模型文件的版本；`--cache-dir` 指向存放 `models--*` 目录的位置，通常为 `$HF_HOME/hub`。

## Python API

```python
from PIL import Image
import trimesh
from sparc3d import Sparc3DPipeline

pipeline = Sparc3DPipeline.from_pretrained(
    "hi3d-official/Hi3Dv1.0",
).to("cuda")

result = pipeline(
    Image.open("input.png"),
    trimesh.load("coarse.glb", force="mesh", process=False),
    resolution=1024,
    num_inference_steps=25,
    seed=0,
)
result.mesh.export("refined.glb")
```

`result` 为 `Sparc3DPipelineOutput`，包含纯几何 `trimesh.Trimesh` 与实际生成的 `resolution`。支持 **512、1024、1536** 三档分辨率，默认每轮进行 25 次调度更新。`seed` 与本地 CPU `torch.Generator` 二选一。若已有形状为 `[N, 3]` 的唯一整数体素坐标，可使用 `pipeline.refine_voxels(image, voxel_coords, structure_resolution=64, resolution=1536, seed=0)`。

### 内存卸载与进度

要减少模型在 GPU 上的常驻内存，请在模型仍位于 CPU 时启用 Accelerate 卸载：

```python
pipeline = Sparc3DPipeline.from_pretrained("hi3d-official/Hi3Dv1.0")
pipeline.enable_sequential_cpu_offload("cuda")
```

该模式替代 `.to("cuda")`，两者不可混用。卸载功能不构成最低显存承诺，也不能消除解码特征的内存开销。

向 `from_pretrained` 传入 `progress_callback`，可在每次调度更新后收到 `Sparc3DProgressEvent`：

```python
def report(event):
    print(f"pass {event.pass_index + 1}/{event.num_passes} "
          f"step {event.step}/{event.num_inference_steps}")

pipeline = Sparc3DPipeline.from_pretrained(
    "hi3d-official/Hi3Dv1.0", progress_callback=report,
).to("cuda")
```

解码、网格提取与后处理阶段不发送调度进度事件。

## 命令行

直接细化已有粗网格，无需 Hunyuan：

```bash
python demo/refine_mesh.py \
  --model hi3d-official/Hi3Dv1.0 \
  --image input.png --coarse-mesh coarse.glb --output refined.glb \
  --resolution 1024 --num-inference-steps 25 --seed 0
```

可添加 `--sequential-cpu-offload` 启用模型卸载；安装演示依赖后，可添加 `--remove-background` 去除背景。去背景功能需主动启用，透明抠图的 alpha 通道会保留用于主体取景。

离线细化时，为三个模型组件提供本地副本：

```bash
python demo/refine_mesh.py \
  --model /path/to/hi3d-v1.0-model \
  --image input.png --coarse-mesh coarse.glb --output refined.glb \
  --local-files-only \
  --dino-model /path/to/dinov2-large \
  --clip-model /path/to/clip-vit-large-patch14
```

## 可选 Hunyuan 粗网格生成示例

`demo/coarse_stage.py` 是 **Hunyuan3D-2 mini** 的可选适配器，使用 `tencent/Hunyuan3D-2mini` 及其 `hunyuan3d-dit-v2-mini-turbo` 子目录。`demo/image_to_mesh.py` 先调用该适配器，再运行细化。Hunyuan 仅用于演示一种上游粗网格生成方式，不是核心库的强制依赖，也不构成产品绑定。可替换适配器，或通过 `demo/refine_mesh.py` 直接提供自己的粗网格；后者没有 Hunyuan 导入。

运行前，需单独安装 [Hunyuan3D-2 代码](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)，并取得其权重：

```bash
python demo/image_to_mesh.py \
  --model hi3d-official/Hi3Dv1.0 \
  --image assets/example.png --output refined.glb \
  --resolution 1024 --seed 0
```

Hunyuan 代码与权重属于第三方发行内容，并非由 Hi3D 开源或重新许可。请遵循[上游源码许可](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE)与 [Hunyuan3D-2 mini 模型许可](https://huggingface.co/tencent/Hunyuan3D-2mini/blob/main/LICENSE)，并参阅[外部模型与可选示例](THIRD_PARTY_LICENSES.md#external-models-and-optional-examples)和[粗网格阶段安装说明](docs/installation.md#the-coarse-stage)。

`--coarse-model` 与 `--coarse-subfolder` 可覆盖默认粗模型。离线运行全流程还需要本地 Hunyuan 模型；细化阶段的 `--local-files-only`、`--revision` 与 `--cache-dir` 不会配置 Hunyuan 加载器。

## 适用范围与可复现性

- **仅几何、单样本。** 不生成纹理、材质或顶点颜色。输出会重新居中并将最大半径归一为 1，不保留输入的世界坐标和绝对尺度。
- **依赖粗结构。** 粗网格中的严重结构错误可能保留；图像不可见区域属于合理补全。
- **需要 GPU 内核。** `flash-attn` 没有替代注意力路径。复杂输入的自适应体素化可能采用更粗的支撑网格，并发出警告；无法达到请求分辨率时会报错，不会静默返回较低分辨率。
- **相同种子不保证相同网格哈希。** 细化采样器使用本地 CPU 随机数生成器，但 GPU 稀疏卷积与注意力并非逐位确定。请在相同输入、模型版本、调度器与环境下比较几何。
- **仅支持推理验证。** 未包含训练代码、评测工具、基准划分与对比网格，不能仅凭本仓库复现论文对比表格。几何细节与标称分辨率也不保证满足特定下游应用。

## 文档

- [安装](docs/installation.md)：已验证环境、Docker、离线加载与 GPU 依赖。
- [流程](docs/pipeline.md)：逐级细化、条件编码、网格提取与内存行为。
- [复现](docs/reproduction.md)：验证范围与有意义的运行间比较。

## 许可证与致谢

Schedule 2 列明的第二阶段实现与修改、文档、配置及八个模型文件中，Hi3D 自有或有权许可的部分适用 **[Hi3D Open License Agreement Version 1.0](LICENSE)**。范围与排除项见 [许可范围](THIRD_PARTY_LICENSES.md#license-scope)，具体以协议正文为准。

- **协议允许商用。** 相关产品或服务的**月活用户超过 200,000**，或**月收入达到 1,000,000 美元**时，须在 **60 日内**通知许可方并申请单独书面许可，同时遵守第 5 条的合并计算与过渡规则。
- **商业集成须展示 `Powered by Hi3D 1.0`**，按第 7 条要求在支持链接的位置链接至 Schedule 2 指定的官网。分发和使用还须遵守协议中的声明要求与可接受使用政策。
- **纯输出适用不同规则。** 获取、编辑、展示、销售或分发不含 Hi3D Materials 或 Model Derivative 的输出，本身不触发上述品牌展示、随附协议／NOTICE 或规模许可义务；适用法律和第三方权利仍须遵守。

原有及第三方部分适用的 **Apache-2.0、MIT 条款继续有效**，其权利不被 Hi3D 协议替换。原 Apache 正文保留于 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md#original-apache-license)，完整第三方条款与归属见 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)、[NOTICE](NOTICE) 和 [AUTHORS](AUTHORS)。本仓库的 Hunyuan 适配示例与外部 Hunyuan 实现、权重及输出相区分，后者仍适用各自条款；DINOv2 与 CLIP 同样适用其自身条款。

Hi3D 许可相关咨询请联系 [contact@hi3d.ai](mailto:contact@hi3d.ai)。

Hi3D v1.0 基于 [Sparc3D: Sparse Representation and Construction for High-Resolution 3D Shapes Modeling](https://arxiv.org/abs/2505.14521) 构建，感谢原始 Sparc3D 工作及相关上游研究。
