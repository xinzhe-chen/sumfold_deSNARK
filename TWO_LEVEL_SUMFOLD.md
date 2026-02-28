# 两级分布式 SumFold 架构重设计

> **状态**: 协议级变更，需形式化安全性分析后方可实施。

## 背景

当前分布式设计中，每个 sub-prover 持有全部 M 个实例（每个实例 N/K 个约束）。这意味着分布式路径并未做实例分区，仅减少了本地实例大小，限制了SumFold的可扩展性。

Benchmark 数据（来自 `deSnark/benches/sumfold_profile.rs`）：

| 场景 | M | nv | 耗时 |
|---|---|---|---|
| 单机 (全量) | 16 | 20 | 5.624s |
| 当前分布式 | 16 | 18 | 596.187ms |
| 真实实例分区 | 4 | 18 | 108.407ms |

## 提议

```
当前: 每个 party 持有 M 个实例, compose_nv = log₂(M) + log₂(N/K)
提议: 每个 party 持有 M/K 个实例
  → 第一层: 各 party 本地 SumFold (log₂(M/K) 轮)
  → 第二层: 跨 party 轻量级 fold 合并 K 个 party 的结果
```

## 为什么这不是实现优化

### 风险 1: Transcript 一致性

当前 `d_sumfold` 的所有 challenge 都从全局 aggregated message 推导（不变量 ⑨）。两级设计中：

- 如果第一层每个 party 独立运行 SumFold（各自 squeeze transcript），则各 party 的 challenge 不同，**无法合并为统一 proof**
- 如果第一层仍需全局同步 challenge，则通信开销与当前设计相同，失去了分区的意义

### 风险 2: 加法可分性被破坏

当前 `d_sumfold` 的关键性质是 prover messages 的加法可分性（不变量 ⑥）：

```
h(b, x) = eq(ρ, b) · Πⱼ f̃ⱼ(b, x)
```

其中 b 的取值范围是全部 M 个实例。如果将 M 分区为 M/K 给每个 party，每个 party 的局部 compose polynomial 只包含 M/K 个实例的贡献。**局部多项式不再是全局多项式的加法分量**，不能简单逐元素求和。

### 风险 3: Folded polynomial 语义不同

单级折叠与两级折叠的数学含义不等价：

```
单级: f̃(x) = Σᵢ₌₁ᴹ eq(r_b, i) · fᵢ(x)
两级: f̃(x) = 第二层fold( Σᵢ∈Pₖ eq(r_b_local, i) · fᵢ(x) )
```

除非精心设计 challenge 的分层结构，两者在一般情况下**不等价**。

### 风险 4: deMKZG 兼容性

`d_multi_open_internal` 期望 folded polynomial 的变量维度为 `num_var`（对应 N/K），party 维度通过 `bit_decompose(party_id)` 嵌入到 opening point 中。两级架构改变了 folded polynomial 的变量结构，需要同步修改 deMKZG 的 opening 协议。

## 可行的研究方向

### 方案 A: Telescoping Transcript

设计分层 transcript 结构：
1. 各 party 独立完成 local SumFold，生成 local proof
2. 将 K 个 local proof 作为输入，运行第二层聚合 SumFold
3. Verifier 先验证第二层 proof，再验证各 local proof

**需要论证**: 两层合成的 soundness 是否等于单层的 soundness。

### 方案 B: Additive Instance Partitioning

保持当前的 compose polynomial 结构，但修改数据分布：
1. 将 M 个实例的 MLE evaluations 分散存储（party k 只存第 k 块）
2. prover round 中每个 party 只计算其本地分片的 partial products_sum
3. 加法可分性自然成立

**需要论证**: 通信模式是否变化、对 `d_prove` 的影响。

### 方案 C: Recursive Composition

将本地 SumFold proof 视为一个新的 NP 语句，用外层证明系统（如 Nova/IVC）递归组合。

**需要论证**: 递归开销是否抵消了分区收益。
