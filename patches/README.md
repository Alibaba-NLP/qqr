# slime Patches

<h4 align="center">
    <p>
        <b>English</b>&nbsp; | &nbsp;
        <a href="README_zh.md">中文</a>
    </p>
</h4>

`slime-rg-sed.patch` contains the slime training-backend changes required by RG-SED
(Reward-Gated Self-Evolution Distillation, referred to as RG-KL in the code), including:

- the `--use-rg-kl` / `--rg-kl-clip-value` arguments;
- passing rollout-side `rg_kl_coef` / `guided_tokens` (the memory-augmented distribution
  conditioned on the teacher prompt) through to the training side;
- `apply_rg_kl_to_advantages`: applies token-level reverse KL on the student's on-policy
  trajectories (teacher distribution under stop-gradient); the coefficient
  lambda_t = lambda_0 * sigmoid((Delta_r - tau)/T) * warmup * decay is precomputed on the rollout side;
- teacher samples get loss_mask=0 and contribute no gradient.

## How to Apply

```bash
cd /path/to/slime
git checkout 8d9378e54aa548a431c00619a19b0dbbdb4f5cd8   # verified base commit
git apply /path/to/qqr/patches/slime-rg-sed.patch
```
