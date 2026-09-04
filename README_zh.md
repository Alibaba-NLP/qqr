# qqr

<h4 align="center">
    <p>
        <a href="README.md">English</a>&nbsp; | &nbsp;
        <b>中文</b>
    </p>
</h4>

<p align="center">
    <img src="assets/Logo.png" width="540"/>
<p>

<p align="center">
    🤗 <a href="https://huggingface.co/collections/Alibaba-NLP/arenarl">HuggingFace</a>&nbsp; | &nbsp;
    🤖 <a href="https://modelscope.cn/datasets/iic/Open-Travel">ModelScope</a>&nbsp; | &nbsp;
    📰 <a href="https://tongyi-agent.github.io/zh/blog/arenarl/">Blog</a>&nbsp; | &nbsp;
    📑 <a href="https://huggingface.co/papers/2601.06487">Paper</a>
<p>

`qqr` 是一个轻量级、非侵入式的 [`slime`](https://github.com/THUDM/slime) 扩展库。集成了 [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol)，通过 **ArenaRL** 算法实现开放域智能体的进化。

## 📰 新闻
- **[2026.08.28]** 🔥 我们发布了 [**ARISE-RL**](https://arxiv.org/abs/2609.01058)（Agentic Rubric-Grounded Iterative Self-Evolution with RL），一个面向开放式智能体的全流程自进化 RL 框架，开源了[出题者/做题者训练代码](qqr/examples/)与专家校准 rubric 基准 [**ECR-Bench**](data/)（ECR-DeepResearch + ECR-Travel）。
- **[2026.07.30]** 🔥 我们发布了 [**SecRespond**](https://arxiv.org/abs/2607.26791)，一个面向真实世界入侵后事件响应的智能体评测基准，并开源了[数据与评测脚本](data/secrespond/)。
- **[2026.05.01]** 🎉 我们的论文 ArenaRL 被 **ICML 2026** 接收！

## 🌟 核心特性

- **ArenaRL 算法**: 完整实现了论文中的核心算法。框架内置了锚点法 (Anchor-Based)、循环赛 (Round-Robin)、瑞士轮 (Swiss-System)、双败淘汰 (Double-Elimination) 和种子单败淘汰制 (Seeded Single-Elimination) 等多种锦标赛拓扑。

- **为开放域智能体设计**: 为解决复杂开放域任务中的判别崩溃问题而设计，即使在奖励模型打分趋于同质化的情况下，依然能通过相对排序驱动策略持续改进。

- **MCP 支持**: 集成 MCP 以标准化本地或远程工具的连接，实现了 LLM 推理与工具环境的解耦。开发者可以直接复用现有的 MCP Servers 作为训练环境，无需重写接口。

- **高性能训练**: 底层基于 [`slime`](https://github.com/THUDM/`) 构建，支持大规模智能体进化所需的高吞吐量分布式生成与训练能力。

## 📦 安装

开始之前，请确保已安装 [`slime`](https://github.com/THUDM/slime)（参考 [快速使用](https://thudm.github.io/slime/zh/get_started/quick_start.html)）。然后通过源码安装 `qqr`：

```bash
git clone https://github.com/Alibaba-NLP/qqr.git
cd qqr
pip install -e .
```

## 🚀 快速开始

通过以下命令启动出行场景的实验：

```bash
bash scripts/travel/run-qwen3-8B.sh
```

您可以在 [`qqr/examples/travel/config.py`](qqr/examples/travel/config.py) 中进行实验相关配置。

## 🧭 ARISE-RL

**ARISE-RL** 通过 rubric 介导的协同演化将任务/rubric **出题者（Generator）**与推理**做题者（Solver）**耦合为一个自进化闭环：出题者把工具相关 rubric 锚定在真实工具观测上，并以难度塑形奖励（`R_G = R_tool · R_fmt · (1 + R_diff)`，三角难度奖励在做题者成功率 ≈ 1/2 处取峰值）生成贴近做题者能力边界的任务；做题者在细粒度 rubric 满足奖励（`r = 0.8·s + 0.2·𝟙[s=1]`）下通过多步推理与工具调用学习。**RG-SED**（Reward-Gated Self-Evolution Distillation）在两个角色上以同一策略 + coach memory 构造临时 teacher，仅当 memory 带来经验奖励提升（Δ_r 通过 sigmoid 门控 `σ((Δ_r−τ)/T)`）时，才施加 token 级 reverse KL 自蒸馏。

### 代码结构

| 目录 | 说明 |
|---|---|
| [`qqr/examples/arise_rl_generator/`](qqr/examples/arise_rl_generator/) | 出题者训练（deepresearch / travel / vitabench） |
| [`qqr/examples/arise_rl_solver/`](qqr/examples/arise_rl_solver/) | 做题者训练（deepresearch / travel / vitabench） |
| [`scripts/arise_rl_generator/`](scripts/arise_rl_generator/)、[`scripts/arise_rl_solver/`](scripts/arise_rl_solver/) | 对应训练脚本 |
| [`data/ecr_deepresearch/`](data/ecr_deepresearch/)、[`data/ecr_travel/`](data/ecr_travel/) | **ECR-Bench**：专家校准 rubric 基准（100 条研究 query / 5×100 条旅行规划 query） |
| [`patches/slime-rg-sed.patch`](patches/) | RG-SED 所需的 slime 训练后端补丁 |

### 快速开始

```bash
# 0. 应用 RG-SED 补丁（详见 patches/README.md）
cd /path/to/slime && git apply /path/to/qqr/patches/slime-rg-sed.patch

# 1. 做题者（Solver）训练，以旅行规划为例
bash scripts/arise_rl_solver/travel/run-qwen3.5-9B.sh

# 2. 出题者（Generator）训练：先启动做题者验证服务，再启动训练
bash scripts/arise_rl_generator/travel/start_executor_server.sh
bash scripts/arise_rl_generator/travel/run-qwen3.5-9B.sh
```

默认超参与论文一致：每 query 采样 G=16 条 rollout（temperature 0.8，至多 40 轮工具交互）；出题者难度奖励采样做题者 K=8 次、成功阈值 γ=0.9；做题者奖励权重 (0.8, 0.2)；RG-SED λ₀=0.5、τ=0.05。所需 API 通过环境变量注入（`DASHSCOPE_API_KEY`、`AMAP_MAPS_API_KEY`、`SEARCH_API_KEY/SEARCH_API_URL` 等），VitaBench 环境数据需从[官方仓库](https://github.com/meituan-longcat/vitabench)获取并设置 `VITABENCH_DATA_DIR`。

## 📋 兼容性

由于 `slime` 的版本升级，特别是涉及 rollout 的改动，请使用 [兼容性](docs/zh/get_started/compatibility.md) 中列出的经过测试的版本组合。

## 致谢

[**slime**](https://github.com/THUDM/slime): 提供了强大的后训练框架。

[**openai-agents-python**](https://github.com/openai/openai-agents-python): 提供了优秀的 MCP 接口。

## 引用

如果您在研究中使用了 `qqr`、ArenaRL 算法或相关基准，请引用我们的论文：

```bibtex
@misc{zhang2026arenarlscalingrlopenended,
      title={ArenaRL: Scaling RL for Open-Ended Agents via Tournament-based Relative Ranking}, 
      author={Qiang Zhang and Boli Chen and Fanrui Zhang and Ruixue Ding and Shihang Wang and Qiuchen Wang and Yinfeng Huang and Haonan Zhang and Rongxiang Zhu and Pengyong Wang and Ailin Ren and Xin Li and Pengjun Xie and Jiawei Liu and Ning Guo and Jingren Zhou and Zheng-Jun Zha},
      year={2026},
      eprint={2601.06487},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.06487}, 
}

@article{wang2026secrespond,
  title={SecRespond: Benchmarking AI Agents for Real-World Post-Compromise Incident Response},
  author={Wang, Lehan and Chen, Boli and Ding, Ruixue and Xie, Pengjun and Huang, Jinwei and Liu, Zhendong and Wang, Shuo and Lei, Tao and Ouyang, Xin and Li, Xiaomeng},
  journal={arXiv preprint arXiv:2607.26791},
  year={2026}
}

@misc{zhang2026ariserl,
  title={ARISE-RL: Agentic Rubric-Grounded Iterative Self-Evolution with Reinforcement Learning},
  author={Zhang, Fanrui and Ding, Ruixue and Zhang, Qiang and Chen, Xi
          and Chen, Boli and Wang, Shihang and Wang, Qiuchen
          and Zhan, Hongmin and Bian, Jinxin and Li, Xingchao
          and Zheng, Peijin and Cheng, Hao and Xie, Pengjun
          and Zhang, Kaipeng and Liu, Jiawei and Zha, Zheng-Jun},
  year={2026},
  eprint={2609.01058},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```
