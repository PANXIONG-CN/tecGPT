# tecGPT Project

Based on GPT-ST: Generative Pre-Training of Spatio-Temporal Graph Neural Networks

## 新增：可插拔模型/数据集（最小侵入）
- 模型插件：`CSA_WTConvLSTM`（目录：`model/CSA_WTConvLSTM/`）
- 数据集插件：`GIMtec`（目录：`lib/datasets/gimtec.py`，数据位于 `data/GIMtec/`）

使用示例（1 epoch 冒烟）：
- `cd model`
- `python Run.py -dataset GIMtec -mode ori -model CSA_WTConvLSTM --epochs 1 --batch_size 2`

更多细节：
- `fix_docs/pluggable_ext.md`（插件机制与落地说明）
- `fix_docs/env_uv.md`（uv 环境配置与运行方式）
