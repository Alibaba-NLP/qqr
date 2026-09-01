# ARISE-RL Generator — VitaBench（交互式生活服务出题者）

<h4 align="center">
    <p>
        <a href="README.md">English</a>&nbsp; | &nbsp;
        <b>中文</b>
    </p>
</h4>

基于 VitaBench 官方 OTA 环境的交互式任务出题者训练。出题者通过真实
OTA 工具探索封闭模拟环境，生成第二人称 `instructions + rubrics`；
做题者以交互模式（Agent ↔ UserSimulator ↔ 工具）验证题目难度。

## 数据与运行

环境与任务数据请从官方仓库
[meituan-longcat/vitabench](https://github.com/meituan-longcat/vitabench) 获取，
并设置 `VITABENCH_DATA_DIR` 指向其 `data/vita` 目录。

```bash
# 1. 启动做题者（Executor）推理服务
bash scripts/arise_rl_generator/vitabench/start_executor_server.sh
# 2. 启动出题者训练（VITABENCH_DOMAIN 可选 ota / delivery / instore / cross_domain）
VITABENCH_DOMAIN=ota bash scripts/arise_rl_generator/vitabench/run-qwen3.5-9B.sh
```

## 场景覆盖（备选 prompt）

出题 system prompt、任务类型与工具链映射均按 `VITABENCH_DOMAIN` 自动切换，
覆盖 VitaBench 全部四个场景：

| 场景 | 任务类型 | 核心工具链 |
|---|---|---|
| `ota`（在线旅行） | 酒店 / 机票 / 火车票 / 景点门票 | `*_search_recommend → get_ota_*_info → create/pay` |
| `delivery`（外卖） | 外卖下单 / 历史复购 / 配送时效 / 订单管理 | `delivery_*_search_recommend → get_delivery_*_info → create/pay_delivery_order`，ETA 换算 |
| `instore`（到店） | 到店团购 / 餐厅订座 / 到店预约 | `instore_*_search_recommend → create/pay_instore_*`、`instore_book`、`instore_reservation` |
| `cross_domain`（跨场景） | 至少覆盖两个域的复合任务 | 上述工具的跨域组合 |

出题种子（`TRAIN_DATA`）每行形如
`{"query": "...", "metadata": {"task_id": "<官方环境 id>", "domain": "<场景>"}}`。

## 奖励设计（论文 Eq.(1)–(5)）

```
R_G = R_tool · R_fmt · (1 + R_diff)
R_tool = 1[出题轨迹包含真实工具观测]
R_fmt  = 1[query 非空 ∧ rubrics 为非空列表]
R_diff = 2 · max(0, 1 − |c − K/2| / (K/2)),  c = Σᵢ 1(rᵢ ≥ γ)
rᵢ     = α·sᵢ + (1−α)·1[sᵢ=1]   （做题者第 i 次试验的奖励）
```

默认超参（与论文一致）：K=8 次做题者验证，γ=0.9，α=0.8。
出题者被引导生成落在做题者能力边界附近（c ≈ K/2）的中等难度任务。

## RG-SED（代码中记为 RG-KL）

三阶段 rollout：student（无 coach memory）→ Coach LLM 生成组级 memory →
teacher（同一策略 + memory，仅用于估计 Δ_r）。
`λ_t = λ_0 · g(Δ_r) · w(t) · d(t)`，其中 `g(Δ_r) = σ((Δ_r − τ)/T)`，
默认 λ_0=0.5、τ=0.05、T=0.0125，cosine warm-up + cosine decay。
仅当 memory 带来经验奖励提升时，才通过 token 级 reverse KL 蒸馏回策略。
