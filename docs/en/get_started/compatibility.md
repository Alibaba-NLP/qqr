# Compatibility

`qqr` is a non-intrusive extension of [`slime`](https://github.com/THUDM/slime): it plugs into slime's rollout and reward interfaces rather than vendoring them. As a result, `slime` version upgrades — in particular refactors of the rollout path — can break `qqr`.

Please use one of the version combinations below, which we have tested to ensure stability.

## Version Matrix

| qqr    | slime  |
| :----- | :----- |
| v0.2.1 | v0.3.1 |
| v0.2.0 | v0.3.0 |
| v0.1.3 | v0.2.4 |
| v0.1.2 | v0.2.3 |
| v0.1.1 | v0.2.2 |
| v0.1.0 | v0.2.1 |

## Installing a Matched Pair

Both projects tag their releases as `vX.Y.Z`, so a tested pair can be pinned by checking out the corresponding tags:

```bash
# slime — install first, refer to its Quick Start for the full setup
git clone https://github.com/THUDM/slime.git
cd slime && git checkout v0.3.1

# qqr
git clone https://github.com/Alibaba-NLP/qqr.git
cd qqr && git checkout v0.2.1 && pip install -e .
```

Refer to slime's [Quick Start](https://thudm.github.io/slime/get_started/quick_start.html) for its complete installation steps.

## Upgrading

When bumping `slime`, upgrade `qqr` to the paired version in the same change. If you need to run a `slime` version that is not listed above, expect to adapt `qqr/rollout/agent_rollout.py` to the new rollout interface.
