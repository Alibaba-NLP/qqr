# qqr

<h4 align="center">
    <p>
        <b>English</b>&nbsp; | &nbsp;
        <a href="README_zh.md">中文</a>
    </p>
</h4>

<p align="center">
    <img src="assets/Logo.png" width="540"/>
<p>

<p align="center">
    🤗 <a href="https://huggingface.co/collections/Alibaba-NLP/arenarl">HuggingFace</a>&nbsp; | &nbsp;
    🤖 <a href="https://modelscope.cn/datasets/iic/Open-Travel">ModelScope</a>&nbsp; | &nbsp;
    📰 <a href="https://tongyi-agent.github.io/blog/arenarl/">Blog</a>&nbsp; | &nbsp;
    📑 <a href="https://huggingface.co/papers/2601.06487">Paper</a>
<p>

`qqr` (a.k.a. hilichurl) is a lightweight, non-intrusive extension for [`slime`](https://github.com/THUDM/slime). It seamlessly integrates the [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) standard to enable the evolution of open-ended agents via [**ArenaRL**](https://arxiv.org/abs/2601.06487).

## 🌟 Key Features

- **ArenaRL Algorithm**: Full implementation of the core algorithms described in the paper. It includes built-in topologies for Anchor-Based, Round-Robin, Swiss-System, Double-Elimination, and Seeded Single-Elimination tournaments.
- **Built for Open-Ended Agents**: Specifically engineered to tackle discriminative collapse in complex, open-ended tasks, ensuring continuous policy improvement via relative ranking even when reward model scores stagnate.
- **MCP Support**: Seamlessly integration with the [MCP]((https://github.com/modelcontextprotocol)) standardizes the decoupling of LLM inference and tool environments. Developers can reuse existing MCP Servers as training environments without rewriting interfaces.
- **High-Performance Training**: Built on top of [`slime`](https://github.com/THUDM/slime) to deliver high-throughput, distributed rollout generation and training for large-scale agent evolution.

## 📦 Installation

To get started, first ensure [`slime`](https://github.com/THUDM/slime) is installed (refer to [Quick Start](https://thudm.github.io/slime/get_started/quick_start.html)). Then install `qqr` from source:

```bash
git clone https://github.com/Alibaba-NLP/qqr.git
cd qqr
pip install -e .
```

## 🚀 Quick Start

Run the travel experiment quickly with the following command:

```bash
bash scripts/travel/run-qwen3-8B.sh
```

You can configure the experiment in [`qqr/examples/travel/config.py`](qqr/examples/travel/config.py).

## 📋 Compatibility

Due to `slime` version upgrades, specifically regarding rollout changes, please use the following version combinations that we have tested to ensure stability:

| qqr    | slime  |
| :----- | :----- |
| v0.1.3 | v0.2.4 |
| v0.1.2 | v0.2.3 |
| v0.1.1 | v0.2.2 |
| v0.1.0 | v0.2.1 |

## Acknowledgements

[**slime**](https://github.com/THUDM/slime): For providing a powerful post-training framework.

[**openai-agents-python**](https://github.com/openai/openai-agents-python): For providing excellent MCP interfaces.

## Citation

If you use `qqr` or the ArenaRL algorithm in your research, please cite our paper:

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