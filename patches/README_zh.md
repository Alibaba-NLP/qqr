# slime 补丁

<h4 align="center">
    <p>
        <a href="README.md">English</a>&nbsp; | &nbsp;
        <b>中文</b>
    </p>
</h4>

`slime-rg-sed.patch` 为 RG-SED（Reward-Gated Self-Evolution Distillation，
代码中记为 RG-KL）所需的 slime 训练后端改动，包括：

- `--use-rg-kl` / `--rg-kl-clip-value` 参数；
- rollout 侧 `rg_kl_coef` / `guided_tokens`（teacher prompt 条件化的 memory-augmented
  分布）透传到训练侧；
- `apply_rg_kl_to_advantages`：对 student on-policy 轨迹施加 token 级 reverse KL
  （teacher 分布 stop-gradient），系数 λ_t = λ_0 · σ((Δ_r − τ)/T) · warmup · decay
  已在 rollout 侧算好；
- teacher 样本 loss_mask=0，不参与梯度。

## 应用方式

```bash
cd /path/to/slime
git checkout 8d9378e54aa548a431c00619a19b0dbbdb4f5cd8   # 已验证的基线 commit
git apply /path/to/qqr/patches/slime-rg-sed.patch
```

### slime v0.3.1 移植版

`slime-rg-sed-v0.3.1.patch` 是 `slime-rg-sed.patch` 向 slime v0.3.1（兼容矩阵中
qqr v0.2.1 指定版本）的移植。原 patch 的验证基线 `8d9378e` 比 v0.3.1 老 243
个提交，已无法直接应用（实测 9 个文件在直接 apply 与 3-way 下全部失败）。

```bash
cd /path/to/slime
git checkout v0.3.1   # a6272da0
git apply /path/to/qqr/patches/slime-rg-sed-v0.3.1.patch
```

关键适配（v0.3.1 在 rollout 侧预计算 DP microbatch 调度）：token 替换在调度
计算之前完成；legacy teacher 前向共享 per-rank schedule 避免分岔死锁；
`apply_rg_kl_to_advantages` 支持 CP（先 `all_gather_with_cp` 对齐再相减，
然后切回本 rank 分块）。已通过 120 rollout 的 GRPO+RG-KL 完整训练验证
（2×8 H20，qwen3.5-9B agent）。
