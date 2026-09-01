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
