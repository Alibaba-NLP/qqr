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

## 📋 兼容性

由于 `slime` 的版本升级，特别是涉及 rollout 的改动，为了确保功能稳定，请使用以下经过测试的版本组合：

| qqr    | slime  |
| :----- | :----- |
| v0.1.2 | v0.2.3 |
| v0.1.1 | v0.2.2 |
| v0.1.0 | v0.2.1 |

## 致谢

[**slime**](https://github.com/THUDM/slime): 提供了强大的后训练框架。

[**openai-agents-python**](https://github.com/openai/openai-agents-python): 提供了优秀的 MCP 接口。

## 引用

如果您在研究中使用了 `qqr` 或 ArenaRL 算法，请引用我们的论文：

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
```