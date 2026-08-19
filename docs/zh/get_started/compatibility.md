# 兼容性

`qqr` 是 [`slime`](https://github.com/THUDM/slime) 的非侵入式扩展：它直接对接 slime 的 rollout 与 reward 接口，而非复制其代码。因此 `slime` 的版本升级（特别是 rollout 链路的重构）可能导致 `qqr` 不可用。

为了确保功能稳定，请使用以下经过我们测试的版本组合。

## 版本对照表

| qqr    | slime  |
| :----- | :----- |
| v0.2.1 | v0.3.1 |
| v0.2.0 | v0.3.0 |
| v0.1.3 | v0.2.4 |
| v0.1.2 | v0.2.3 |
| v0.1.1 | v0.2.2 |
| v0.1.0 | v0.2.1 |

## 安装匹配的版本组合

两个项目的发布版本均以 `vX.Y.Z` 形式打 tag，因此可以通过切换到对应 tag 来锁定已测试的版本组合：

```bash
# slime —— 请先安装，完整步骤参考其快速使用文档
git clone https://github.com/THUDM/slime.git
cd slime && git checkout v0.3.1

# qqr
git clone https://github.com/Alibaba-NLP/qqr.git
cd qqr && git checkout v0.2.1 && pip install -e .
```

slime 的完整安装步骤请参考其 [快速使用](https://thudm.github.io/slime/zh/get_started/quick_start.html) 文档。

## 版本升级

升级 `slime` 时，请在同一次变更中将 `qqr` 一并升级到配对的版本。如果需要使用上表未列出的 `slime` 版本，通常需要将 `qqr/rollout/agent_rollout.py` 适配到新的 rollout 接口。
