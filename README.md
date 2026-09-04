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

## 📰 News
- **[2026.08.28]** 🔥 We release [**ARISE-RL**](https://arxiv.org/abs/2609.01058) (Agentic Rubric-Grounded Iterative Self-Evolution with RL), a full-cycle self-evolution RL framework for open-ended agents, together with the [Generator/Solver training code](qqr/examples/) and the expert-calibrated rubric benchmark suite [**ECR-Bench**](data/) (ECR-DeepResearch + ECR-Travel).
- **[2026.07.30]** 🔥 We release [**SecRespond**](https://arxiv.org/abs/2607.26791), a benchmark for real-world post-compromise incident response, together with its [dataset and evaluation scripts](data/secrespond/).
- **[2026.05.01]** 🎉 Our paper ArenaRL has been accepted by **ICML 2026**!

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

## 🧭 ARISE-RL

**ARISE-RL** couples a task/rubric **Generator** and a reasoning **Solver** into a rubric-mediated self-evolution loop. The Generator grounds tool-related rubric criteria in real tool observations and is trained with a difficulty-shaped reward (`R_G = R_tool · R_fmt · (1 + R_diff)`, a triangular reward peaking when the Solver succeeds on about half of its attempts), so that generated tasks track the Solver's capability boundary. The Solver learns from fine-grained rubric satisfaction rewards (`r = 0.8·s + 0.2·𝟙[s=1]`) through multi-step reasoning and tool use. **RG-SED** (Reward-Gated Self-Evolution Distillation) builds a transient teacher from the *same* policy conditioned on coach memory, and applies token-level reverse KL self-distillation only when the memory empirically improves reward (sigmoid gate `σ((Δ_r−τ)/T)`).

### Layout

| Path | Description |
|---|---|
| [`qqr/examples/arise_rl_generator/`](qqr/examples/arise_rl_generator/) | Generator training (deepresearch / travel / vitabench) |
| [`qqr/examples/arise_rl_solver/`](qqr/examples/arise_rl_solver/) | Solver training (deepresearch / travel / vitabench) |
| [`scripts/arise_rl_generator/`](scripts/arise_rl_generator/), [`scripts/arise_rl_solver/`](scripts/arise_rl_solver/) | Training scripts |
| [`data/ecr_deepresearch/`](data/ecr_deepresearch/), [`data/ecr_travel/`](data/ecr_travel/) | **ECR-Bench**: expert-calibrated rubric benchmarks (100 research queries / 5×100 travel-planning queries) |
| [`patches/slime-rg-sed.patch`](patches/) | slime backend patch required by RG-SED |

### Quick Start

```bash
# 0. Apply the RG-SED patch (see patches/README.md)
cd /path/to/slime && git apply /path/to/qqr/patches/slime-rg-sed.patch

# 1. Solver training, e.g. travel planning
bash scripts/arise_rl_solver/travel/run-qwen3.5-9B.sh

# 2. Generator training: start the Solver (executor) service first, then train
bash scripts/arise_rl_generator/travel/start_executor_server.sh
bash scripts/arise_rl_generator/travel/run-qwen3.5-9B.sh
```

Default hyperparameters follow the paper: G=16 rollouts per query (temperature 0.8, up to 40 rounds of tool interaction); K=8 Solver rollouts with success threshold γ=0.9 for the difficulty-shaped Generator reward; Solver reward weighting (0.8, 0.2); RG-SED λ₀=0.5, τ=0.05. Required API credentials are injected via environment variables (`DASHSCOPE_API_KEY`, `AMAP_MAPS_API_KEY`, `SEARCH_API_KEY`/`SEARCH_API_URL`, ...). VitaBench environment data should be obtained from the [official repository](https://github.com/meituan-longcat/vitabench) with `VITABENCH_DATA_DIR` set accordingly.

## 📋 Compatibility

Due to `slime` version upgrades, specifically regarding rollout changes, please use one of the tested version combinations listed in [Compatibility](docs/en/get_started/compatibility.md).

## Acknowledgements

[**slime**](https://github.com/THUDM/slime): For providing a powerful post-training framework.

[**openai-agents-python**](https://github.com/openai/openai-agents-python): For providing excellent MCP interfaces.

## Citation
 
If you use `qqr` or the ArenaRL algorithm in your research, please cite our papers:

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
