# sumfold_deSNARK 修改建议

## 目标

这份文档面向当前仓库的主项目 `sumfold_deSNARK`。

- `sumfold_deSNARK`：主项目，本次建议的主体
- `HyperPlonk`：主项目的单机基座和参考实现，不是对比实验对象
- `HyperPianist`：对比实验对象

当前代码更准确的定位是：

> 一个结构清晰、可跑 benchmark、具有研究价值的分布式证明原型。

它距离“协议完整、验证闭环完整、可以严肃宣称为完成态 deSNARK”的状态，还有几步关键修改。

---

## 先说结论

如果你现在的目标是：

- 做内部性能调优
- 看 `nv / M / K` 的扩展趋势
- 对比你当前实现和 `HyperPianist` 的工程代价

那么仓库已经有相当高的研究原型价值。

如果你的目标是：

- 给出完整的协议正确性主张
- 给出可对外发布的 verifier 接口
- 支撑论文级别的安全性和 benchmark 结论

那么需要优先修掉下面几类问题：

1. 最终 verifier 没有真正验证 SumFold 阶段
2. verifier 没有把固定电路语句绑定到 verifying key
3. Fiat-Shamir transcript 没有完整绑定 commitments
4. 当前 setup 是 testing-only trusted setup

---

## HyperPlonk 在仓库中的作用

`HyperPlonk` 在这个仓库里承担的是“单机语义基座”的作用，主要提供：

- 电路和索引结构：`MockCircuit`、`HyperPlonkIndex`、`HyperPlonkParams`
- proving / verifying key 类型：`HyperPlonkProvingKey`、`HyperPlonkVerifyingKey`
- gate polynomial 构造与求值：`build_f`、`eval_f`
- 预处理逻辑：`preprocess`

`deSnark` 在此基础上做的是：

- 复用 `HyperPlonk` 的电路语义和预处理
- 把 proving 主路径改造成 `SumFold + distributed SumCheck + distributed PCS`
- 保留 `HyperPlonk` 作为电路/语义/类型系统的上游来源

因此，建议的总体方向不是“去掉 HyperPlonk”，而是：

- 明确 `HyperPlonk` 是 base layer
- 明确 `deSnark` 是 distributed proving layer
- 明确两者的协议边界和 transcript 边界

---

## 修改建议总表

### P0：必须先改的

1. ✅ [DONE] 让最终 verifier 真正验证 SumFold
2. ✅ [DONE] 用 VK 绑定固定电路，而不是信任 proof 里自带的 selector commitments
3. ✅ [DONE] 统一并补完整个 Fiat-Shamir transcript 的承诺绑定
4. ✅ [DONE] 把 testing-only setup 和真实协议 setup 明确分离

### P1：应该尽快改的

5. 明确 `deSnark` 的协议范围：只证明 gate 约束，还是完整替代 HyperPlonk
6. 重新定义并修正文档中的 benchmark 口径
7. 让分布式 workload 语义更清晰，避免“看起来分布式，实则复制式 shard”带来的误解
8. 增加 K>1 的端到端自动化测试

### P2：工程质量改进

9. 降低网络层 panic / unchecked deserialize 风险
10. 清理重复逻辑和临时分支，收敛主路径
11. 提升脚本的可移植性和结果复现性

---

## P0-1：让最终 verifier 真正验证 SumFold

### 当前问题

现在的正式验证主路径里，`SumFold` 没有作为一个独立、被完整检查的阶段进入最终 verifier。

现状更像是：

- 先重放 `SumFold` transcript，得到同样的 challenge
- 再用这些 challenge 去验证后面的 HyperPianist SumCheck

但这不等价于“验证了 SumFold proof 本身”。

特别是：

- `proof.sum_t` 没有在最终 verifier 中形成闭环检查
- `merge_and_verify_sumfold(...)` 只在 `debug_assertions` 下辅助检查
- 正式 verifier 里 `SumFold` 更像 challenge 前缀，而不是被验证的协议阶段

### 应该改成什么

最终 verifier 至少要显式做两步：

1. 先验证 `SumFold` 阶段，得到：
   - `rho`
   - `r_b`
   - `c`
2. 再验证后续 HyperPianist / folded polynomial 阶段

### 建议实现方式

新增一个真正的对外 verifier 入口，例如：

```rust
pub fn verify<E: Pairing>(
    vk: &VerifyingKey<E, MultilinearKzgPCS<E>>,
    proof: &Proof<E, MultilinearKzgPCS<E>>,
    public_statement: ...
) -> Result<bool>
```

然后把验证拆成两个显式阶段：

#### 阶段 A：验证 SumFold

- 调用 `verify_sum_fold(...)` 或整理一个 `verify_sum_fold_with_transcript(...)` 的正式封装
- 检查：
  - `proof.num_sumfold_rounds == proof.q_aux_info.num_variables`
  - `subclaim.point == proof.proof.point[..num_sumfold_rounds]`
  - `subclaim.expected_evaluation == proof.v * eq(rho, r_b)`
  - `sum_t` 的一致性

#### 阶段 B：验证 folded polynomial 的 SumCheck / PCS

- 在阶段 A 成功之后，再继续验证后半段 proof
- 明确后半段的 asserted sum 应该是什么，以及它与阶段 A 输出的关系

### 最少要补的负例测试

- 篡改 `sum_t` 应失败
- 篡改任意一轮 SumFold `prover msg` 应失败
- 篡改 `r_b` 的任意一位应失败
- 交换两轮 SumFold 证明消息应失败

### 预期收益

- verifier 变成真正的协议验证器，而不是 prover-side self-check
- 论文或文档中可以更明确地声称”SumFold 已被正式验证”

### ✅ 实施记录

- **文件**: `deSnark/src/snark.rs` (`verify_proof_eval` 函数)
- **改法**: 将被动 transcript replay（仅重放 append/squeeze 操作）替换为调用 `verify_sum_fold_with_transcript`，该函数在执行完全相同的 transcript 操作的同时，还执行 `verify_round_and_update_state` 和 `check_and_generate_subclaim` 进行完整的 round 验证
- **新增检查**: SumFold subclaim 一致性检查 `c == v * eq(ρ, r_b)`
- **`r_b` 来源**: 现在使用 `sf_subclaim.point`（从验证中得出），替代直接从 proof 中提取
- **transcript 兼容性**: `verify_sum_fold_with_transcript` 的 transcript 操作序列与原有 replay 完全一致，后续 HyperPianist 验证不受影响

---

## P0-2：用 verifying key 绑定固定电路

### 当前问题

`HyperPlonk` 的设计里，固定电路语句应由 verifying key 绑定。

但 `deSnark` 当前 PCS 路径中，verifier 使用的是 `proof.selector_commits`，而不是 `vk.selector_commitments`。
这会导致 verifier 实际验证的是：

> “存在一组自洽的 selector + witness commitments”

而不是：

> “存在满足 verifying key 所指定固定电路的 witness”

这在语义上是不同的。

### 应该改成什么

对于固定 selector：

- verifier 应直接使用 `vk.selector_commitments`
- proof 不应再主导 selector commitment 的语义

proof 里如果保留 `selector_commits`，也只能用于：

- 调试
- 或与 `vk.selector_commitments` 做相等性断言

不能把它当成 verifier 的信任输入。

### 建议改法

#### 方案 A：最干净

- 从 `Proof` 里删除 `selector_commits`
- verifier 始终只使用 `vk.selector_commitments`

#### 方案 B：兼容性更强

- 保留 `selector_commits`
- verifier 中强制检查：

```rust
proof.selector_commits == vk.selector_commitments
```

- 然后后续 PCS batch verify 仍然只用 `vk.selector_commitments`

### 还要顺手检查的点

- `proof.witness_commits` 是 prover 自带输入，这没有问题
- `selector_commitments` 应被视作 statement / VK 的组成部分，而不是 witness 的组成部分

### 必补测试

- 把 proof 里的 selector commitments 替换成另一组值，应失败
- 在 VK 不变的情况下，替换 proof 的 selector commitments 为”自洽但不同”的值，应失败

### ✅ 实施记录

- **文件**: `deSnark/src/snark.rs` (`verify_proof_eval` 函数)
- **改法**: 采用方案 A+B 结合：
  1. 显式检查 `proof.selector_commits` 与 `vk.selector_commitments` 数量和逐个相等
  2. PCS batch_verify 中使用 `vk.selector_commitments[j]` 而非 `sel_commits[j]` 进行 folded commitment 重构
- **效果**: 若 proof 篡改 selector commitments，会在相等性检查处直接拒绝；即使绕过检查，PCS 验证也使用 VK 值

---

## P0-3：统一并补完整个 Fiat-Shamir transcript

### 当前问题

当前 transcript 有两个核心缺口：

1. 上层 `SumFold + d_prove` 的主 transcript 在生成 challenge 前，没有完整吸收 selector / witness commitments
2. PCS opening 又启用了一个新的 `pcs_transcript`

这会造成一个典型问题：

- IOP 的随机挑战没有先绑定到底层 commitments
- PCS 的 opening transcript 又和主 transcript 脱开了

从协议设计上，这会让“证明对象”和“挑战来源”的绑定变弱。

### 应该改成什么

建议把 transcript 设计成一条明确的主链，或者至少是“主链派生子链”。

#### 推荐方案：单主 transcript

顺序建议如下：

1. append circuit/VK 相关固定语句
2. append selector commitments
3. append witness commitments
4. 开始 SumFold transcript
5. 开始 folded polynomial 的 SumCheck transcript
6. 进入 PCS opening / batch opening transcript

#### 次优方案：主 transcript 派生 PCS transcript

如果你坚持保留独立 `pcs_transcript`，建议：

- 先在主 transcript 中吸收所有 commitments 和前序 proof 状态
- 再从主 transcript squeeze 一个 domain-separated seed
- 用这个 seed 初始化 PCS transcript

这样至少不是完全脱链。

### 建议落地步骤

1. 先写一份“协议 transcript 调度表”
2. 代码实现完全对齐这份调度表
3. prover 和 verifier 都按同一个调度表重放

### 文档里应该出现的内容

需要明确写出：

- 哪些字段进入 transcript
- 进入顺序是什么
- 每一步 challenge 的 label 是什么
- 哪些 transcript 是主 transcript，哪些是派生 transcript

### ✅ 实施记录

- **文件**: `deSnark/src/snark.rs`
- **主 transcript 改动**:
  - Prover (`dist_prove_sumcheck`): K=1 和 K>1 两条路径中，transcript 初始化后、SumFold 开始前，均吸收 `vk.selector_commitments`（label: `b"sel_cm"`）
  - Verifier (`verify_proof_eval`): 同样在 transcript 初始化后吸收 `vk.selector_commitments`
  - 函数签名增加 `sel_commits: &[Commitment<E>]` 参数
- **PCS transcript 改动**:
  - Prover 和 Verifier 两端均在 PCS transcript 初始化后、追加 folded commits 前，先吸收 `sum_t` 和 `v`（label: `b"sum_t"`, `b"v"`），将 PCS transcript 绑定到主协议状态
- **当前 transcript 调度表**:
  1. 主 transcript: `sel_cm` × N → `aux info` → `sumfold rho` → per-round SumFold → per-round HyperPianist
  2. PCS transcript: `sum_t` → `v` → `pcs_cm` × (num_sel+num_wit) → batch_verify 内部

---

## P0-4：把 testing-only setup 和真实 setup 分开

### 当前问题

`setup()` 目前直接使用：

- `test_rng()`
- `gen_srs_for_testing()`

这在 benchmark 和单元测试里没问题，但在协议语义上它不应叫“正式 setup”。

### 应该改成什么

把 setup 分成两层：

#### 层 1：bench/test setup

例如：

```rust
pub fn setup_for_bench(...)
```

特点：

- deterministic
- 可缓存
- 明确标注 unsafe / testing only

#### 层 2：real setup / imported CRS

例如：

```rust
pub fn load_or_validate_srs(...)
pub fn setup_from_trusted_source(...)
```

特点：

- 不使用 `test_rng`
- 语义上是真正的 CRS 输入

### 你至少应该做到的

即使暂时不做真实 setup，也建议：

- 改名
- 加 feature gate
- 在 README 中明确写“当前仓库 benchmark 使用 testing-only SRS，不构成真实 trusted setup”

这样不会把 bench 辅助逻辑误包装成协议正式部分。

---

## P1-5：明确协议范围

### 当前问题

现在 `deSnark` 更像是在证明：

- HyperPlonk gate polynomial 的 folded / distributed 版本

但还没有严格完成：

- public input 语义闭环
- permutation / copy constraints 的完整 distributed 版本

这本身不一定是 bug，但必须写清楚，否则对外表达会模糊。

### 两种可选方向

#### 方向 A：承认当前范围

把项目定位成：

> 基于 HyperPlonk gate polynomial 的 distributed proving prototype

这样你只需要：

- 文档写清楚范围
- benchmark 解释清楚 workload
- verifier 闭环做好

#### 方向 B：把协议补全

目标是：

> 完整 distributed HyperPlonk / deSNARK

那就要继续补：

- public input 绑定
- permutation / copy constraints 的 distributed 化
- 与 VK 绑定的固定语句

### 我的建议

短期建议先走方向 A，把协议边界写清楚，把 verifier 和 transcript 修好。
中期再看是否继续补全完整 distributed HyperPlonk 语义。

---

## P1-6：修正文档和 benchmark 口径

### 当前问题

文档和脚本里有几处口径不一致：

- `nv` / `n` / `2^n` 的描述混用
- `setup_ms` 是否包含 SRS / circuit generation 的说法不完全一致
- `avg_cpu_pct` 和 `peak_rss_mb` 很容易被误读成“全系统总资源”
- localhost 多进程结果容易被误读成真实跨机结果

### 应该改成什么

#### 定义一个统一术语表

- `nv = log_num_constraints`
- `N = 2^nv`
- `M = 2^log_num_instances`
- `K = 2^log_num_parties`

#### 在 README 中明确写出

- `setup_ms` 是否包括 SRS 生成
- `setup_ms` 是否包括 circuit generation
- `prover_ms / verifier_ms` 是否做了 `/ M`
- `avg_cpu_pct` 是 master-only 还是 cluster aggregate
- `peak_rss_mb` 是单进程峰值还是所有进程的峰值

#### 对 HyperPianist 对比要加一句硬说明

> 这是一组工程对比 benchmark，不是严格协议等价成本对比。

### 建议新增的 benchmark 输出列

如果你想更严谨：

- `cluster_comm_sent`
- `cluster_comm_recv`
- `master_cpu_pct`
- `worker_cpu_pct_sum`
- `master_peak_rss_mb`
- `sum_peak_rss_mb`
- `localhost_mode=true/false`

---

## P1-7：明确分布式 workload 语义

### 当前问题

当前默认跑通的是 constraint-distribution 路径，而 instance-distribution 被显式禁用。
这没有问题，但要避免文档把它描述成“通用 distributed proving”。

尤其是：

- 当前各 party 的本地电路生成方式非常接近“同构 shard”
- benchmark 更像在测当前实现的分布式开销和缩放，而不是通用异构电路划分

### 应该改成什么

在文档里明确 workload 是哪一种：

#### 方案 A：保留当前模型

定义为：

> replicated-structure, constraint-sharded proving benchmark

也就是：

- 固定结构相同
- 局部 witness 独立
- 通信 / folding / opening 的代价可被测出

#### 方案 B：改成更真实的 party-specific shard

如果你想更像真实分布式场景，需要：

- 每个 party 的 shard 明确由 `party_id` 决定
- 输入数据和 witness 生成与 party 绑定
- statement 层面说明“所有 party shard 共同组成全局实例”

### 关于 instance-distribution

你的代码已经说明它目前在乘积结构上不成立，那就继续禁用是对的。

建议：

- 保留禁用状态
- 在文档里明确说“当前不支持”
- 后续要恢复时，先补 algebraic correctness 说明，再补代码

---

## P1-8：增加真正的 K>1 自动化测试

### 当前问题

现在测试更多是：

- 单机
- K=1
- 或逻辑等价模拟

这对算法开发足够，但对分布式协议不够。

### 应该补哪些测试

#### 集成测试

- 2-party localhost end-to-end
- 4-party localhost end-to-end
- 小参数下完整 prove + verify roundtrip

#### 负例测试

- worker 返回错误 prover message
- master 篡改聚合消息
- 篡改 folded commitment
- 篡改 batch opening

#### 一致性测试

- K=1 分布式路径结果与单机路径一致
- transcript challenge 序列与预期一致
- folded eval 与本地直接 eval 一致

#### benchmark sanity test

- 小规模参数下 benchmark 至少能稳定跑完一轮
- CSV 头和列数固定

### 测试建议

尽量把“真实网络多进程测试”单独放成 integration test 或 script-driven CI job，
不要只依赖 `cargo test --workspace`。

---

## P2-9：网络层硬化

### 当前问题

网络层现在更偏研究型实现，有这些典型特征：

- 大量 `unwrap()`
- 大量 `deserialize_*_unchecked`
- shutdown 依赖固定 sleep
- receiver thread 没有正常 join

这在 benchmark 阶段可以接受，但不适合协议层长期依赖。

### 建议修改

1. 把 `deserialize_uncompressed_unchecked` 改成受检反序列化，至少在外部输入边界受检
2. 给网络消息增加更明确的错误路径，而不是 panic
3. 给 `deinit()` 一个真正的线程退出信号
4. 移除固定 `sleep(3s)` 这种收尾方式
5. 如果 `channel_id` 不打算真的支持多路并发，就删掉多余抽象；如果打算支持，就把调用链补齐

### 修改优先级

这是工程稳定性问题，不如 verifier 和 transcript 严重，但会直接影响长期维护成本。

---

## P2-10：清理重复逻辑，收敛主路径

### 当前问题

`make_circuit()` 和 `dist_prove()` 里都在做一套类似的：

- preprocess
- trim PCS params
- 电路构造
- selector / witness 准备

主路径和 bench 路径之间也有一些重复口径。

### 建议修改

抽出更清晰的层次：

#### 层 1：circuit preparation

- build circuits
- preprocess
- produce pk/vk

#### 层 2：statement preparation

- build selector/witness MLEs
- build sumcheck instances

#### 层 3：proof generation

- d_commit
- d_sumfold
- d_prove
- commitment folding
- d_multi_open

#### 层 4：verification

- verify sumfold
- verify folded sumcheck
- verify PCS

这样代码会更容易做：

- bench 复用
- integration test 复用
- 后续协议切换

---

## P2-11：脚本与复现性改进

### 当前问题

脚本目前偏 Linux/WSL 假设：

- `bash`
- `lsof`
- `pkill`

另外 bench 结果目录、日志目录、SRS 文件目录虽然已经比较清晰，但还可以更规范。

### 建议修改

1. 给 `scripts/` 目录补一份“支持环境说明”
2. 把 benchmark 输出文件名和参数保持完全同构
3. CSV 文件旁边增加一份 metadata 文件，记录：
   - git commit
   - toolchain
   - host CPU
   - RAYON threads
   - localhost / multi-host
4. 明确把 benchmark 分成：
   - internal benchmark
   - publishable comparison benchmark

---

## 建议的落地顺序

### 第 1 阶段：先把协议闭环补上

目标：

- 有正式 verifier
- verifier 真正检查 SumFold
- verifier 使用 VK 的固定 selector commitments
- transcript 调度明确且一致

完成后，项目就从“原型可跑”变成“原型但验证闭环完整”。

### 第 2 阶段：把安全语义和文档口径补齐

目标：

- setup 明确分层
- README 明确协议范围
- benchmark 口径统一
- 对 HyperPianist 的对比表述更严谨

完成后，项目就能更安全地对外表达。

### 第 3 阶段：补 K>1 自动化测试和网络稳定性

目标：

- 小参数多进程 CI
- 负例测试
- 网络退出与错误处理改进

完成后，项目的维护成本会显著下降。

### 第 4 阶段：再决定要不要补全完整 distributed HyperPlonk 语义

这一步是架构决策：

- 如果只做 gate-folding 原型，到这里已经足够强
- 如果要做完整 distributed SNARK，再继续补 permutation / public input 语义

---

## 一个更现实的项目定位建议

如果你近期要继续写文档、答辩或汇报，我建议你的表述从：

> 我实现了一个完整的 sumfold_deSNARK

调整为：

> 我实现了一个基于 HyperPlonk 语义基座的 distributed proving prototype，
> 其核心贡献是将多实例 gate polynomial 通过 SumFold 折叠，并与 distributed SumCheck / PCS 结合，
> 当前重点是 proving pipeline 和 benchmark，对完整 verifier 闭环与严格语义绑定仍在继续完善。

这会更稳，也更符合当前代码状态。

---

## 最终评价

### 优点

- 模块划分清楚
- 核心思路明确
- 注释质量整体不错
- benchmark 管线已经具备研究原型价值
- `HyperPlonk -> deSnark` 的复用关系比较清楚

### 核心短板

- 协议验证闭环还没完全收口
- statement / VK / proof 的边界还不够严格
- benchmark 可用，但还不够“可直接支撑强结论”

### 代码水平判断

如果按研究原型标准：

- 属于中上水平

如果按完整密码协议实现标准：

- 当前还在“有扎实基础，但关键安全和验证层未完工”的阶段

---

## 供后续执行时参考的文件优先级

建议后续改动优先看这些文件：

- `deSnark/src/snark.rs`
- `deSnark/src/d_sumfold.rs`
- `deSnark/src/structs.rs`
- `hyperplonk/src/utils.rs`
- `hyperplonk/src/structs.rs`
- `subroutines/src/poly_iop/sum_check/mod.rs`
- `subroutines/src/pcs/deMultilinear_kzg/deMkzg.rs`
- `deNetwork/src/channel.rs`
- `deNetwork/src/multi.rs`
- `README.md`
- `deSnark/README.md`
- `deSnark/examples/dist_bench.rs`
- `scripts/run_interactive_bench.sh`

