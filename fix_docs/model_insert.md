 在保持 `conf/lib/model` 目录与 Run/Trainer 流程不变的前提下，引入“接口 + 注册表 + 适配器”三件套，实现 LLM 与自定义时空模型的可插拔扩展，改动仅限两处入口与新增若干文件，兼容原有模型与配置。

------

## 架构与依赖表

> 依据仓库现状梳理职责与耦合，标注可扩展点/脆弱耦合点。

| 层级/目录  | 关键文件/模块                              | 主要职责                                              | 上游依赖                                            | 下游依赖                                  | 可扩展点                                  | 脆弱耦合点                                                 |
| ---------- | ------------------------------------------ | ----------------------------------------------------- | --------------------------------------------------- | ----------------------------------------- | ----------------------------------------- | ---------------------------------------------------------- |
| **驱动**   | `model/Run.py`                             | 解析参数、构造数据、创建模型与 Trainer、启动训练/评测 | `lib/Params_pretrain.py`、`lib/Params_predictor.py` | `model/Model.py`、`model/BasicTrainer.py` | ✅ 新增插件配置入口（可选）                | 与参数解析强绑定（预训练/预测双体系）                      |
| **训练**   | `model/BasicTrainer.py`                    | 通用 train/val/test、日志、早停、保存                 | 损失函数、模型 `forward`                            | 日志、保存                                | –                                         | `forward` 返回值约定（五元组占位）较硬；指标形状假设固定。 |
| **集成**   | `model/Model.py`                           | 预训练增强（GPT-ST）与下游预测器拼装                  | `model/Pretrain_model/GPTST.py`                     | 各子模型（STGCN/MTGNN…）                  | ✅ 预测器构造改为注册表工厂                | 以 `if-elif` 选择模型，新增模型需改源。                    |
| **参数**   | `lib/Params_pretrain.py`                   | 读取 GPT-ST 预训练 INI                                | `conf/GPTST_pretrain/*.conf`                        | `Run.py`                                  | –                                         | –                                                          |
| **参数**   | `lib/Params_predictor.py`                  | 为各下游模型装配参数                                  | 各模型 `args.py`                                    | `Run.py`                                  | ✅ 加入“解析器注册表”与插件 YAML/JSON 合并 | 同样以 `if-elif` 分发。                                    |
| **数据**   | `lib/dataloader.py`、`lib/load_dataset.py` | 切窗、标准化、拆分                                    | `conf/*/*.conf`                                     | `Run.py`                                  | –                                         | 返回形状/标准化与 `input_base_dim` 强耦合。                |
| **模型集** | `model/*/*`                                | 各基线实现与 `args.py`                                | –                                                   | `model/Model.py`                          | –                                         | 子模型输入/输出张量排列不一，靠集成层适配。                |

> 结论：将“预测器选择”和“参数解析”两个硬编码分发点抽象为**注册表**，通过**适配层**统一张量形状与生命周期，即可最小侵入地引入 LLM 与自定义 ST 模型。

------

## 方案设计

### 设计目标

- **最小改动**：仅在 `model/Model.py` 与 `lib/Params_predictor.py` 加两段“尝试走注册表，否则走原 if-elif”的逻辑；其余均为新增文件。
- **统一接口**：`ModelInterface` 规范 `forward/train_step/save/load/validate_config`，但对现有模型仅要求 `forward`（保留 Trainer 负责训练循环）。
- **配置驱动**：在不破坏 INI 的基础上，**可选**增加 `--plugin_config` (YAML/JSON)，合并为命名空间属性。
- **依赖反转**：`Model.py` 不再直接 import 新模型，转而向 `MODEL_REGISTRY` 请求构造；各插件自注册。

### 关键组件

1. **接口**：`model/interfaces.py`

- 约束最小通用能力；为旧模型提供默认空实现，不强迫迁移。

1. **注册表+工厂**：`model/registry.py`

- `MODEL_REGISTRY.register(name)(builder)`：注册构造器（返回 `nn.Module`/`ModelInterface` 实例）。
- `ARGS_REGISTRY.register(name)(arg_parser)`：注册参数解析器（补充/覆盖 `--plugin_config`）。
- `build(name, *args)`：按名构造，不存在则抛 `KeyError`。

1. **适配层（两类）**

- **LLMAdapter（HF）**：`model/adapters/llm_hf.py`
  - 以 `inputs_embeds` 方式调用 HF 模型（如 GPT‑2），不引入 tokenizer 复杂性；将 `[B,N,D,T]` 展平为 `[B,T,N*D]`，投影到隐藏维度，经 LLM 后用 `Linear` 变换为 `[B,H,out_dim*N]` 再 reshape。* 依赖可懒加载。
- **STAdapter（自定义）**：`model/adapters/st_head.py`
  - 轻量 TCN(时间卷积) + 可学习邻接（线性“伪”GCN）头；统一输出为 `[B,H,out_dim,N]`，满足 Trainer/Metric 假设。

1. **配置合并**：`lib/config_ext.py`

- 若传入 `--plugin_config path`，支持 YAML/JSON 读取并将键值写回 `args`/`args_predictor`（不存在的键自动新增）。

### 两套落地路径

- **A. 不重构版（推荐；最小改动）**
  - `Model.py` 与 `Params_predictor.py` 在原 if-elif 前**优先尝试注册表**；找不到再回退。
  - 原有 `.conf`/`args.py` 全保留，命令行不变。
- **B. 轻重构版**
  - 替换两处 if-elif 为纯注册表；将各旧模型以“适配器+自注册”方式迁入（可渐进式进行）；新增一个统一 `args_base.py` 供模型参数共享。

------

## 变更清单（新增/修改/删除）

> 仅列出必要最小改动；其余为新增文件。

### 新增

```
model/interfaces.py
model/registry.py
model/adapters/llm_hf.py
model/adapters/st_head.py
model/LLM_HF/args.py            # LLM 插件专用参数解析
conf/plugins/llm_gpt2.yaml      # LLM 示例配置（YAML）
conf/plugins/st_custom.yaml     # 自定义时空模型示例配置
lib/config_ext.py               # 插件配置合并工具
```

### 修改（行/函数级锚点）

1. **`model/Model.py` → `Enhance_model.__init__`（“预测器创建”附近）**
    **在原 if-elif 分发之前插入（搜索锚点：`if self.model == 'MTGNN':` 之上）：**

```python
# [新增] 优先尝试从注册表构建（若无则回退到原有 if-elif）
from .registry import MODEL_REGISTRY
built = False
try:
    self.predictor = MODEL_REGISTRY.build(self.model, args_predictor, args.device, dim_in, dim_out)
    built = True
except KeyError:
    built = False

if not built:
    # 原有 if-elif 分发保持不变
    ...
```

> 仅新增 ~10 行，不改动既有分支；所有旧模型无感。

1. **`lib/Params_predictor.py` → `get_predictor_params`**
    **在创建 `parser_pred` 后增加：**

```python
# [新增] 通用插件配置与注册表解析
parser_pred.add_argument('--plugin_config', default=None, type=str)
try:
    from model.registry import ARGS_REGISTRY
    args_predictor = ARGS_REGISTRY.parse(args.model, args.dataset, parser_pred, outer_args=args)
    return args_predictor
except Exception:
    pass  # 回退至原有 if-elif
```

**在函数末尾（return 前）合并 YAML/JSON（也可在 registry 里做）：**

```python
from lib.config_ext import maybe_merge_plugin_config
args_predictor = maybe_merge_plugin_config(args_predictor)
return args_predictor
```

> 原分发逻辑保留，插件优先解析，可选配置合并。

### 删除

- 无

------

## 代码片段（含 ModelInterface、注册表、两个示例适配器、示例配置）

> **说明**：以下代码为可运行骨架（省略 import 注释）；与仓库 PyTorch 风格一致。LLM 适配器基于 HuggingFace `transformers`，为**可选依赖**，采用懒加载设计。

### `model/interfaces.py`

```python
import torch
import torch.nn as nn
from typing import Any, Dict, Optional

class ModelInterface(nn.Module):
    """统一接口：最小化约束，兼容现有 Trainer 只需 forward。"""
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 [B, N, D_in, T_in]，输出 [B, H, D_out, N]。"""
        raise NotImplementedError

    # 可选：供插件使用，旧模型默认空实现
    def train_step(self, batch: Any) -> Dict[str, Any]:
        return {}

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None) -> None:
        self.load_state_dict(torch.load(path, map_location=map_location))

    @staticmethod
    def validate_config(cfg: Any) -> None:
        """可在这里做参数校验（维度、horizon 合法性等）。"""
        return
```

------

## 新增模型：TEC_MoLLM（Vendor 复制/最小侵入集成）

参考《OptimizedCSA_WTConvLSTM_workflow.md》中的“年段切分、边界补帧、流式 stride=1”的数据流程，本仓库以最小改动接入 Vendor 模型 TEC_MoLLM：

- 目录与文件
  - `model/TEC_MoLLM/tec_mollm.py`：轻量适配器，封装 Vendor 实现为统一预测器接口（forward 只接收 `x`）。
  - `model/TEC_MoLLM/args.py`：解析 `conf/TEC_MoLLM/GIMtec.conf`，提供 heads/patch_len/d_llm 等模型超参。
  - `conf/TEC_MoLLM/GIMtec.conf`：数据与模型默认配置，保持与 Vendor 一致的初值。
  - `lib/dataloader.get_dataloader`：新增 `model_key == 'tec_mollm'` 分支，GIMtec/TEC 数据默认走“年段+边界补帧+流式 stride=1”。

- 数据流关键点
  - 自动年段切分：沿用 `lib/datasets/gimtec_pretrain.build_gimtec_pretrain_dataloaders` 的划分（2009–2012|2014|2016|2018 训练；2013、2017 验证；2015、2019–2022 测试）。
  - 边界补帧：各验证/测试段默认在段首拼接上游段的 `lag` 帧作为 warm-up（prefix_boundary=True）。
  - 流式+stride=1：窗口化以 stride=1 产生样本，DataLoader 不一次性物化全部窗口，避免巨量内存占用。

- 图结构（edge_index）
  - 统一从 `lib/datasets/gimtec_adj.load_or_build_adj` 获取（优先 data/adj/<DATASET>/ 最新；否则自动构建 71×73 八邻接并保存）。
  - 适配器内将稠密 A 转成 `edge_index`，并常驻 GPU 缓存以降低重复开销。

- 推理与保存
  - `val` 默认不落盘数组；`test` 默认保存 `preds.npy`（float16），不再保存 ground-truth；`test` 自动生成按年 JSON 指标，路径与日志同名。

- GPU/LLM 资源
  - 若已安装 `transformers` 且本地缓存了 `gpt2`（如 `/root/autodl-tmp/cache`），适配器将优先使用 HF 模型；否则降级到轻量 TinyTransformer，仍完整走 GPU 图前向。

使用示例（标准入口 · 冒烟 + 测试）：

```
cd model
# 训练 1 epoch
HF_HOME=/root/autodl-tmp/cache HF_HUB_OFFLINE=1 \
python Run.py -dataset GIMtec -mode ori -model TEC_MoLLM -epochs 1 -batch_size 32 -amp True

# 完整测试导出（读取 best_model.pth 软链）
python Run.py -dataset GIMtec -mode test -model TEC_MoLLM -batch_size 32 -amp True
```

说明：
- 仅通过 `model/Run.py` 训练/验证/测试（不再使用 Vendor 的 Run 脚本）。
- Vendor 源码已复制到 `model/TEC_MoLLM/vendor/` 并做本地化导入；去除了任何 `/root/VendorCode` 绝对路径依赖。
- GIMtec 的时间分辨率设为 2 小时（`interval_minutes=120`）；`lag=12,horizon=12` 表示“过去 24 小时 → 未来 24 小时”。


### `model/registry.py`

```python
from typing import Callable, Dict, Any
from argparse import ArgumentParser
import importlib

class _ModelRegistry:
    def __init__(self):
        self._builders: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def deco(fn: Callable[..., Any]):
            self._builders[name] = fn
            return fn
        return deco

    def build(self, name: str, *args, **kwargs):
        if name not in self._builders:
            raise KeyError(f"model '{name}' not registered")
        return self._builders[name](*args, **kwargs)

MODEL_REGISTRY = _ModelRegistry()

class _ArgsRegistry:
    def __init__(self):
        self._parsers: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str):
        def deco(fn: Callable[..., Any]):
            self._parsers[name] = fn
            return fn
        return deco

    def parse(self, name: str, dataset: str, parser: ArgumentParser, outer_args=None):
        if name not in self._parsers:
            raise KeyError(f"args parser for '{name}' not registered")
        return self._parsers[name](dataset, parser, outer_args)

ARGS_REGISTRY = _ArgsRegistry()
```

### `lib/config_ext.py`

```python
import json, os
from typing import Any
try:
    import yaml
except Exception:
    yaml = None

def _load(path: str):
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yml", ".yaml")):
            if yaml is None:
                raise RuntimeError("PyYAML not installed")
            return yaml.safe_load(f)
        return json.load(f)

def maybe_merge_plugin_config(args: Any) -> Any:
    path = getattr(args, "plugin_config", None)
    if not path:
        return args
    cfg = _load(path) or {}
    for k, v in cfg.items():
        setattr(args, k, v)
    return args
```

### `model/adapters/llm_hf.py`（LLM 插件）

```python
import torch
import torch.nn as nn
from . import hf_utils  # 可选：如需拆分懒加载逻辑
from ..interfaces import ModelInterface
from ..registry import MODEL_REGISTRY, ARGS_REGISTRY

# -------- 注册参数解析器 --------
@ARGS_REGISTRY.register("LLM_HF")
def parse_args_llm(dataset, parser, outer_args=None):
    parser.add_argument('--hf_model_name', type=str, default='gpt2')
    parser.add_argument('--llm_hidden', type=int, default=768)  # 若离线初始化
    parser.add_argument('--freeze_backbone', type=bool, default=True)
    parser.add_argument('--horizon', type=int, default=getattr(outer_args, 'horizon', 12))
    parser.add_argument('--use_pretrained', type=bool, default=True)
    # 允许 --plugin_config 覆盖
    args, _ = parser.parse_known_args()
    return args

class HFLLMAdapter(ModelInterface):
    """将 [B,N,D,T] 经线性投影-> LLM(inputs_embeds)-> 线性头 -> [B,H,D_out,N]"""
    def __init__(self, args, device, dim_in, dim_out):
        super().__init__()
        self.device = device
        self.dim_in, self.dim_out = dim_in, dim_out
        self.horizon = getattr(args, 'horizon', 12)
        self.num_nodes = getattr(args, 'num_nodes', None)

        # 懒加载 HF，以免成为硬依赖
        try:
            from transformers import AutoConfig, AutoModel
            if getattr(args, 'use_pretrained', True):
                cfg = AutoConfig.from_pretrained(args.hf_model_name)
                self.backbone = AutoModel.from_pretrained(args.hf_model_name)
            else:
                cfg = AutoConfig.from_pretrained(args.hf_model_name)
                self.backbone = AutoModel(cfg)
            hidden = cfg.hidden_size
        except Exception:
            # 无 transformers 时退化为纯 MLP
            hidden = getattr(args, 'llm_hidden', 768)
            self.backbone = nn.Identity()

        in_feat = (self.num_nodes or 1) * self.dim_in
        out_feat = (self.num_nodes or 1) * self.dim_out
        self.in_proj = nn.Linear(in_feat, hidden)
        self.head = nn.Linear(hidden, out_feat)

        if hasattr(self.backbone, 'parameters') and getattr(args, 'freeze_backbone', True):
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,N,D,T]
        B, N, Din, T = x.shape
        x_seq = x.permute(0, 3, 1, 2).reshape(B, T, N * Din)  # [B,T,N*D]
        embeds = self.in_proj(x_seq)                           # [B,T,H]
        if hasattr(self.backbone, 'forward'):
            out = self.backbone(inputs_embeds=embeds).last_hidden_state  # [B,T,H]
        else:
            out = embeds
        out = out[:, -self.horizon:, :]                        # 取最后 H 步
        y = self.head(out).reshape(B, self.horizon, self.dim_out, N)  # [B,H,D_out,N]
        return y

# -------- 注册构造器 --------
@MODEL_REGISTRY.register("LLM_HF")
def build_llm_hf(args_predictor, device, dim_in, dim_out):
    return HFLLMAdapter(args_predictor, device, dim_in, dim_out)
```

### `model/adapters/st_head.py`（自定义时空模型插件）

```python
import torch
import torch.nn as nn
from ..interfaces import ModelInterface
from ..registry import MODEL_REGISTRY, ARGS_REGISTRY

@ARGS_REGISTRY.register("ST_CUSTOM")
def parse_args_st(dataset, parser, outer_args=None):
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--horizon', type=int, default=getattr(outer_args, 'horizon', 12))
    args, _ = parser.parse_known_args()
    return args

class SimpleSTHead(ModelInterface):
    """TCN + 可学习邻接(线性)；输入 [B,N,D,T] 输出 [B,H,D_out,N]。"""
    def __init__(self, args, device, dim_in, dim_out):
        super().__init__()
        self.horizon = getattr(args, 'horizon', 12)
        self.dim_out = dim_out
        self.num_nodes = getattr(args, 'num_nodes', 1)
        hidden = getattr(args, 'hidden', 64)

        # 时间卷积（按 node 独立）
        self.tcn = nn.Conv1d(dim_in, hidden, kernel_size=3, padding=1)
        # 可学习“邻接” -> 线性混合节点
        self.A = nn.Parameter(torch.eye(self.num_nodes))
        # 读出未来 H 步
        self.readout = nn.Conv1d(hidden, self.horizon * dim_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,N,D,T]
        B, N, Din, T = x.shape
        x = x.reshape(B * N, Din, T)
        h = torch.relu(self.tcn(x))                            # [B*N, hidden, T]
        y = self.readout(h)                                    # [B*N, H*D_out, T]
        y = y[..., -1]                                         # 简化：用最后时刻的卷积输出
        y = y.reshape(B, N, self.horizon, self.dim_out)        # [B,N,H,D_out]
        # 空间线性混合
        y = torch.einsum('b n h d, n m -> b m h d', y, self.A) # [B,M(H=同N),H,D]
        return y.permute(0, 2, 3, 1)                           # [B,H,D_out,N]

@MODEL_REGISTRY.register("ST_CUSTOM")
def build_st_head(args_predictor, device, dim_in, dim_out):
    return SimpleSTHead(args_predictor, device, dim_in, dim_out)
```

### `model/LLM_HF/args.py`（备用：若你更偏好与旧模型一致的 `args.py` 形式）

```python
# 可选：给 LLM 插件单独提供 "args.py"，亦可完全依赖 registry 的解析器。
def parse_args(DATASET, parser, outer_args=None):
    parser.add_argument('--hf_model_name', type=str, default='gpt2')
    parser.add_argument('--freeze_backbone', type=bool, default=True)
    args, _ = parser.parse_known_args()
    return args
```

### 示例插件配置（YAML）

```
conf/plugins/llm_gpt2.yaml
hf_model_name: "gpt2"
freeze_backbone: true
horizon: 12
conf/plugins/st_custom.yaml
hidden: 64
horizon: 12
```

### 命令行入口（保持原流程）

```bash
# 以 LLM（HF）作为预测器，在 NYC_TAXI 上跑 ori 流程
python model/Run.py -dataset NYC_TAXI -mode ori -model LLM_HF --plugin_config ../conf/plugins/llm_gpt2.yaml

# 以自定义 ST 头作为预测器
python model/Run.py -dataset PEMS08 -mode ori -model ST_CUSTOM --plugin_config ../conf/plugins/st_custom.yaml
```

> 原有命令不受影响，例如：
>  `python model/Run.py -dataset PEMS08 -mode eval -model STGCN` 仍按原分支执行。

------

## 测试与验收

**单测（pytest）**

1. **注册表**：注册/构造成功与异常（未注册抛 `KeyError`）。
2. **接口契约**：两类适配器传入伪数据 `x: [2, N, D, T]`，输出维度严格等于 `[2, H, D_out, N]`。
3. **配置合并**：`--plugin_config` 覆盖命令行/INI；YAML/JSON 均通过。

**集成测试**

1. **E2E（无 HF）**：卸载 `transformers`，`LLM_HF` 自动退化为 MLP，能完成一个 epoch。
2. **E2E（有 HF）**：安装 `transformers` 后，`LLM_HF` 以 `inputs_embeds` 跑通 1~3 个 batch。
3. **回归**：对任意原模型（如 STGCN/GWN），有无注册表改动，**日志/指标一致**（浮点抖动范围内）。

**性能/基线对齐**

- 对旧模型：训练时长、显存峰值偏差 < 1%。
- 对新模型：提供**对齐标准**
  - 形状正确、梯度非 NaN；
  - `BasicTrainer.test` 中 MAE/RMSE/Mape 能随训练下降；
  - 5 个时间步平均 MAE 较“恒等预测/历史复制”基线至少提升 5%。

------

## 迁移与兼容

- **启用方式**：新增模型只需：实现适配器 → 在 `MODEL_REGISTRY`/`ARGS_REGISTRY` 注册 →（可选）提供 YAML。旧命令无需变化。
- **回退策略**：若注册表未找到或插件导入失败，自动回退到原 if‑elif ；HF 缺失时 LLM 退化为 MLP，不影响流程。
- **预训练增强**：如需让 LLM 参与 `mode='eval'`（即 GPT‑ST 增强），仅需保证适配器 `forward` 接受增强后的嵌入（`Model.py` 已统一在 `Enhance_model` 中处理）。

------

## 风险清单与建议

1. **形状/维度不一致**：不同基线张量排列差异较大 → 由适配器**统一输出为 `[B,H,D_out,N]`**；在 `validate_config` 校验 `horizon/out_dim/num_nodes`。
2. **HF 依赖体积/版本**：`transformers` 与 `torch` 版本耦合 → 采用**懒加载**与**可选依赖**，并在 README 中标注建议版本（如 torch≥1.9 与相应 HF 版本）。
3. **显存与速度**：LLM `inputs_embeds` 会放大内存 → 在 YAML 中暴露 `hf_model_name`（可用 `gpt2`/`distilgpt2`）与冻结选项，默认冻结主干仅训练头部。
4. **日志与保存**：Trainer 的保存路径依赖 `args.log_dir` 与 `save_pretrain_path` → 插件复用 Trainer 的 `save()`/`load()`，不自定义保存位置。
5. **设备设置脆弱**：`TrainInits.init_device` 通过 `int(opt.device[5])` 解析 CUDA id，格式依赖 `'cuda:0'` → 建议未来轻重构版改为 `torch.device` 解析（本次不动）。
6. **数据标准化耦合**：`lib/dataloader.normalize_dataset` 对 `input_base_dim`/day/week 拆分有强假设 → 自定义模型如不需要额外时间特征，在适配器内仅使用前 `D_in`。

------

### 补充：如何新增一个你的自定义模型（两步走）

1. 在 `model/adapters/your_model.py` 中实现 `ModelInterface` 并注册：

```python
@ARGS_REGISTRY.register("MY_MODEL")
def parse_args_my(dataset, parser, outer_args=None):
    parser.add_argument('--hidden', type=int, default=128)
    args, _ = parser.parse_known_args()
    return args

@MODEL_REGISTRY.register("MY_MODEL")
def build_my(args_predictor, device, dim_in, dim_out):
    return MyModelAdapter(args_predictor, device, dim_in, dim_out)
```

1. 运行：

```bash
python model/Run.py -dataset METR_LA -mode ori -model MY_MODEL --plugin_config ../conf/plugins/my_model.yaml
```

> 无需改任何原有文件（除本方案已做的两处一次性改动）。

------

**说明**：以上方案严格保持原目录与核心职责边界（Run/Trainer/数据/原模型均不变），通过最小注入实现“可插拔模型”扩展，既可纳入 HuggingFace LLM，也可快速接入自定义时空模型。

---

# TEC_MoLLM for GIMtec 接入指引（Vendor 复制 · 最小侵入）

本节为新增模型 VendorCode/CSA-WTConvLSTM/Models/TEC_MoLLM 的接入说明，严格参考《OptimizedCSA_WTConvLSTM_workflow.md》的年段切分、边界补帧与流式窗口（stride=1）流程，实现一致的评估与导出规范，同时尽可能复用现有代码。

要点约定（本次 smoke 生效）：
- 输出目录固定：`/root/tecGPT/Output/GIMtec/TEC_MoLLM/`（不再创建下层子目录）。
- run_name 命名：`YYYYmmdd_HHMMSS_GIMtec_TEC_MoLLM_<mode>`（含 mode 后缀）。
- 保存策略：
  - 验证集（val）不保存数组；
  - 测试集（test）仅保存 `<run_name>_test_preds.npy`（float16），不保存 true；
  - 同时生成 `<run_name>_test.json`（字段风格与 OCSAWT 一致）。
- 滑窗与切分：训练/验证/测试统一 `stride=1`；年段内窗口不跨年，验证/测试段在段首拼接上一段 `lag` 帧作为边界补帧（prefix）。
- GPT‑2：固定使用 `gpt2-large` 离线加载，缓存目录 `/root/autodl-tmp/cache/`，冻结参数；`llm_layers=6`；`d_llm` 自动从 HF config 读取（`config.n_embd=1280`）。
- 融合：默认“加性融合（与 Vendor 一致）”；提供 LayerNorm 开关（默认关闭），位置在“加性后、temporal 前”。
- 本次 smoke 仅使用 TEC 基础通道（C=1）；后续再与 OCSAWT 完全一致地引入 day/week 两个额外通道（C=3）。

## 数据管线对齐 OCSAWT（年段 + 边界补帧 + 流式 stride=1）

复用 `lib/datasets/gimtec_pretrain.py` 的核心逻辑：

1）年段组织与边界前缀（引用片段）

```python
# lib/datasets/gimtec_pretrain.py（节选）
# 验证段：2013 以前缀来自 2012 段；2017 以前缀来自 2016 段
va_2013 = pack_years([2013])
va_2017 = pack_years([2017])
if prefix_boundary:
    tr_2009_2012 = tr_segments[0]
    va_2013 = np.concatenate([tr_2009_2012[-args.lag:], va_2013], axis=0)
    tr_2016 = tr_segments[2]
    va_2017 = np.concatenate([tr_2016[-args.lag:], va_2017], axis=0)

# 测试段：按 vendor 级联前缀
te_2015 = pack_years([2015])
te_2019 = pack_years([2019])
te_2020 = pack_years([2020])
te_2021 = pack_years([2021])
te_2022 = pack_years([2022])
if prefix_boundary:
    tr_2014 = tr_segments[1]
    tr_2018 = tr_segments[3]
    te_2015 = np.concatenate([tr_2014[-args.lag:], te_2015], axis=0)
    te_2019 = np.concatenate([tr_2018[-args.lag:], te_2019], axis=0)
    te_2020 = np.concatenate([te_2019[-args.lag:], te_2020], axis=0)
    te_2021 = np.concatenate([te_2020[-args.lag:], te_2021], axis=0)
    te_2022 = np.concatenate([te_2021[-args.lag:], te_2022], axis=0)
```

2）统一滑窗 stride=1 与流式 Dataset（引用片段）

```python
# lib/datasets/gimtec_pretrain.py（节选）
class _SegmentWindowDataset(Dataset):
    def __init__(self, seg, lag, horizon, single, scaler_data, scaler_day, scaler_week, input_base_dim, stride=1):
        ...
        self.stride = max(1, int(stride))
        T = seg.shape[0]
        end = max(0, T - (lag + horizon) + 1)
        self.length = 0 if end <= 0 else (end + self.stride - 1) // self.stride
    def __getitem__(self, idx):
        s = idx * self.stride
        e_x = s + self.lag
        ...
        x = self.seg[s:e_x]
        y = self.seg[s_y:e_y]
        # 组分缩放（base/day/week），本次 smoke 仅用 base 通道
        xb = self.scaler_data.transform(x[..., :self.input_base_dim])
        yb = self.scaler_data.transform(y[..., :self.input_base_dim])
        ...
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
```

DataLoader 建议（与 OCSAWT 保持一致）：`pin_memory=True`、`persistent_workers=True`、`num_workers=8~16`，测试/验证关闭 shuffle。

## 评估与导出（对齐 OCSAWT：val 不落数组；test 落 preds.npy + JSON）

1）仅保存预测数组（float16，不保存 true），引用片段：

```python
# model/BasicTrainer.py::test（节选）
base_stem = os.path.splitext(log_filename)[0]
np.save(os.path.join(args.log_dir, f"{base_stem}_{save_tag}_preds.npy"),
        y_pred.detach().cpu().to(torch.float16).numpy())
# true 不落盘（可由数据与窗口规则重建）
```

2）JSON 指标结构复用 OCSAWT（overall + per_year），引用片段：

```python
# lib/eval_gimtec_yearwise.py::compute_yearwise_metrics（节选）
overall = {
  'mse_norm': 0.0,
  'rmse_norm': 0.0,
  'rmse_real_TECU': 0.0,
  'mae_real_TECU': 0.0,
  'relative_error_percent': 0.0,
  'count_frames': 0,
  'shape_per_frame': [1, args.height, args.width],
  'per_year': { '2015': {...}, '2019': {...}, ... }
}
```

本模型导出规范：
- 目录：`/root/tecGPT/Output/GIMtec/TEC_MoLLM/`
- 文件：
  - 日志：`<run_name>.log`
  - 预测：`<run_name>_test_preds.npy`（float16）
  - JSON：`<run_name>_test.json`

## 模型与融合（Vendor 为准，最小改）

1）GPT‑2 加载（默认 gpt2-large，离线缓存；冻结参数）

```python
# /root/VendorCode/CSA-WTConvLSTM/Models/TEC_MoLLM/modules.py（现状节选）
from transformers import AutoModel
...
base = AutoModel.from_pretrained('gpt2')
# 仅保留前几层（llm_layers）
base.h = nn.ModuleList(list(base.h)[:layers])
```

建议的“最小改”实现（示意）：

```python
# 支持 gpt2-large + 本地缓存 + 冻结
name = os.environ.get('HF_MODEL_NAME', 'gpt2-large')
cache_dir = os.environ.get('HF_HOME', '/root/autodl-tmp/cache')
base = AutoModel.from_pretrained(name, cache_dir=cache_dir, local_files_only=True)
# 自动 d_llm = base.config.n_embd
for p in base.parameters():
    p.requires_grad_(False)
```

2）融合策略与 LayerNorm 开关（默认关闭）

```python
# /root/VendorCode/CSA-WTConvLSTM/Models/TEC_MoLLM/tec_mollm.py（现状节选）
s = self.spatial(x_4g, edge_index)  # [L*B, N, Cs]
# 维度可对齐时做加性融合（与 Vendor 一致）
if s.size(-1) % C == 0:
    s = s + x_4g.repeat(1, 1, s.size(-1) // C)
# [可选] LayerNorm 开关：加性后、temporal 前提升稳定性（默认关闭）
# if self.use_ln:
#     s = s.view(L, B, N, Cs).permute(0, 1, 3, 2)
#     s = self.ln(s).permute(0, 1, 3, 2).reshape(L*B, N, Cs)
```

LayerNorm 的作用：
- 对齐与稳定跨模态（LLM/时空编码）特征尺度，缓解协变量偏移；
- 在 A800 + AMP(bf16/fp16) 下提升数值稳定性与可控性；
- 默认关闭以保持与 Vendor 结果一致，可按需要打开。

## A800 / AMP / TF32 建议

```bash
export NVIDIA_TF32_OVERRIDE=1
# 代码内：
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# AMP：优先 bfloat16；指标以 float32 聚合；不兼容自动回退 fp16
```

DataLoader：`num_workers=8~16`、`pin_memory=True`、`persistent_workers=True`、`prefetch_factor` 适度；迁移 `non_blocking=True`；若主干以 2D 卷积为主可启用 `channels_last`。

## 冒烟运行（1 epoch，单卡 A800）

本次 smoke 使用单通道（C=1）、`llm_layers=6`、batch_size=32（若 OOM 自动降至 16→8）。

```bash
# 严格离线的 HF 缓存
export HF_HOME=/root/autodl-tmp/cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NVIDIA_TF32_OVERRIDE=1

# 训练 + 验证（1 epoch）与测试导出（Vendor 入口）
python /root/VendorCode/CSA-WTConvLSTM/Run_TEC_MoLLM.py \
  --data-path /root/tecGPT/data/GIMtec \
  --output-dir /root/tecGPT/Output/GIMtec/TEC_MoLLM/ \
  --epochs 1 --batch-size 32 \
  --train-stride 1 --val-stride 1 --test-stride 1 \
  --llm-layers 6
```

产出示例（均在 `/root/tecGPT/Output/GIMtec/TEC_MoLLM/`）：
- 日志：`20250915_112233_GIMtec_TEC_MoLLM_train.log`
- 预测：`20250915_112233_GIMtec_TEC_MoLLM_test_preds.npy`
- JSON：`20250915_112233_GIMtec_TEC_MoLLM_test.json`

## 与 OCSAWT 对齐要点清单
- 年段划分：训练（2009–2012、2014、2016、2018）、验证（2013、2017）、测试（2015、2019–2022）。
- 边界补帧：验证/测试段在段首拼接上一段 `lag` 帧（prefix）。
- 滑窗：训练/验证/测试统一 `stride=1`，不跨年。
- 指标与 JSON：字段与计算逻辑复用 OCSAWT；val 不保存数组；test 不保存 true。
- 日志与命名：run_name 模板一致；文件命名风格一致（`<run_name>_test.json`）。
- 输出目录：固定 `/root/tecGPT/Output/GIMtec/TEC_MoLLM/`。

---

## 标准入口整合与日志/保存策略（新增）

- 本地 vendor 导入：
  - `model/TEC_MoLLM/tec_mollm.py`
    ```python
    # 改为本地 vendor 引用
    from model.TEC_MoLLM.vendor.tec_mollm import TEC_MoLLM as _VendorTEC
    self.model = _VendorTEC(..., use_ln=args_predictor.use_ln,
                            hf_model_name=args_predictor.hf_model_name,
                            hf_cache_dir=args_predictor.hf_cache_dir)
    ```
- interval=120（仅 GIMtec/TEC）：
  - `model/Run.py` 在构造 DataLoader 前注入 `args.interval=120`；`lib/datasets/gimtec_pretrain.py` 默认也回落到 120。
- ckpt 策略（全模型统一）：
  - 训练保存 `run_name.pth`，并创建软链 `best_model.pth -> run_name.pth`；日志打印两者。
  - `model/BasicTrainer.py::save_checkpoint` 已实现统一策略，非 TEC_MoLLM 亦适用。
- test JSON 的 meta（仅 GIMtec/TEC 注入）：
  - `model/BasicTrainer.py` 在写 JSON 前注入：
    ```json
    {
      "meta": {
        "run_name": "YYYYmmdd_HHMMSS_GIMtec_TEC_MoLLM_test",
        "paths": {"preds": "..._test_preds.npy", "log": "..._test.log", "ckpt": "..._train.pth"},
        "env": {"amp": "bf16", "tf32": true, "ln_enabled": false},
        "oom_fallback": {"initial_batch_size": 32, "final_batch_size": 32, "events": []},
        "model": {"gpt2_name": "gpt2-large", "llm_layers": 6, "frozen": true},
        "data": {"dataset": "GIMtec", "channels": 1, "stride": 1, "interval_minutes": 120, "lag": 12, "horizon": 12}
      }
    }
    ```

## 开启 LayerNorm 的推荐时机（新增）

- 以下情况建议 `use_ln=True`（加性融合后、temporal 卷积前）：
  - 提高 `batch_size`、`llm_layers`（如 6→12）、`temporal_channels`、`spatial_out/heads` 时出现训练不稳或梯度振荡；
  - 从 bf16 切换到 fp16、或显存/吞吐压力较大导致数值敏感时；
  - 观测到 loss/指标异常波动时作为稳定器启用。
- 默认关闭以保持 Vendor 复现路径；开启后可在日志与 JSON meta 中看到 `ln_enabled=true`。

## 目录清理说明（新增）

- 旧目录 `/root/tecGPT/Models` 已移除，避免与 `model/TEC_MoLLM` 重复冲突；请统一在 `model/` 下查看与维护模型实现。
