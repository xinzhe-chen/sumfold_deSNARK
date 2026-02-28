# 安全优化方案

项目 1-5 已实施并通过全部 34 项测试验证。项目 6-8 暂缓（需更深入的 pipeline 改造）。

---

## 1. `fix_variables` → `fix_variables_in_place`

**来源**: Report #1, Report #4, NOTES #4

**现状**: `sum_fold_v2` 每轮 SumCheck round 对 t 个 MLE 调用 `fix_variables`，每次 `vec!` 分配新内存。`eq_fix` 也同理。

**优化**: 切换为 `arithmetic/src/multilinear_polynomial.rs` 中已实现的 `fix_variables_in_place`，覆写前半段，零堆分配。

**安全性论证**: 计算的是完全相同的线性插值 `f'(x₂,...,xₙ) = (1-r)·f(0,x₂,...) + r·f(1,x₂,...)`。只改变内存布局，不改变计算结果。`d_sumfold.rs` 中已对 `eq_fix_evals` 使用了此函数，验证了等价性。

**涉及文件**:
- `subroutines/src/poly_iop/sum_check/mod.rs` — `sum_fold_v2` L839-847, L847
- `deSnark/src/d_sumfold.rs` — L196-203 (compose_mle 的手动降维可进一步改为 in-place)

---

## 2. `compose_mle_evals` 构建并行化

**来源**: Report #2

**现状**: 三重循环 `for j in 0..t { for x.. { for i..m } }` 完全顺序执行。

**优化**: 外层 `into_par_iter()` 或预分配 + `par_chunks_mut()` 并行写入。

**安全性论证**: `compose_mle[j][x·m + i] = polys[i].mle[j][x]` 是纯数据转置，各元素写入完全独立，无数据依赖。并行化不改变输出内容。

**涉及文件**:
- `subroutines/src/poly_iop/sum_check/mod.rs` — `sum_fold_v1` L495-509, `sum_fold_v2` L778-792, `sum_fold_v3` L1090-1098
- `deSnark/src/d_sumfold.rs` — L144-153

---

## 3. 取模 → 位与掩码 (`%` → `&`)

**来源**: Report #3

**现状**: 热点内循环中 `acc[0][b % (1 << (length - round - 1))]` 使用取模运算。

**优化**: 预计算 `let mask = (1 << (length - round - 1)) - 1;`，替换为 `acc[0][b & mask]`。

**安全性论证**: 当 `n = 1 << k` 时，`b % n ≡ b & (n-1)` 对所有非负整数 b 成立。`1 << X` 恒为 2 的幂，等价性有数学保证。

**涉及文件**:
- `subroutines/src/poly_iop/sum_check/mod.rs` — `sum_fold_v1` L613+L617, `sum_fold_v2` L895+L899, `sum_fold_v3` L1183+L1187
- `deSnark/src/d_sumfold.rs` — L249+L252

---

## 4. 生产路径从 `sum_fold_v2` 切换到 `sum_fold_v3`

**来源**: NOTES #2

**现状**: `prove_sumfold()` 使用 `sum_fold_v2`，debug 模式下交叉验证 v1/v2/v3 一致性。

**优化**: 将主路径切换为 `sum_fold_v3`。

**安全性论证**: v3 与 v2 共享完全相同的协议结构：
- 相同的 transcript 操作序列
- 相同的 compose polynomial 构造
- 相同的 `sum_t` / `v` / `proof` 输出

`prove_sumfold()` L395-434 的 debug 断言已验证三版本输出完全一致（`sum_t`、`v`、`proof`、`aux_info`、`folded_poly`）。

**涉及文件**:
- `deSnark/src/snark.rs` — `prove_sumfold()` L362-364

---

## 5. 移除 `hyperplonk` debug 默认 features

**来源**: NOTES #3

**现状**: `hyperplonk/Cargo.toml` 默认启用 `extensive_sanity_checks` 和 `print-trace`。

**优化**: 默认只保留 `parallel`，debug features 改为显式启用。

**安全性论证**: 这些 features 只控制 `assert!` 和 `tracing` 调用的编译时插入，不参与任何密码学计算。

**涉及文件**:
- `hyperplonk/Cargo.toml`

---

## 6. MLE 复用（消除 pipeline 中的重复构建）

**来源**: NOTES #4

**现状**: Phase 1.5a 构建 selector/witness MLEs 用于 `d_commit`，Phase 1.5b 的 `circuits_to_sumcheck()` 又从原始数据重建一遍。

**优化**: 构建一次，通过 `Arc<DenseMultilinearExtension>` 共享。

**安全性论证**: 相同输入数据构建的 MLE 内容完全一致。`Arc::make_mut` 在引用计数 > 1 时自动 clone，保证共享安全。

**涉及文件**:
- `deSnark/src/snark.rs` — `circuits_to_sumcheck()`, d_commit 相关代码
- `hyperplonk/src/snark.rs` — `mul_prove()` 中的 witness clone

---

## 7. PCS 批处理点去重

**来源**: NOTES #6

**现状**: 外部构造 `vec![same_point.clone(); folded_total]` 传入批处理层，内部再去重。

**优化**: 在构造阶段提前去重，避免大量重复 `Vec<F>` 的分配。

**安全性论证**: `d_multi_open_internal` 内部已有去重逻辑（L92-107）。提前去重只是显式化了内部的隐式操作，commitment 同态 `C(g') = Σ [eq(a₂,pᵢ)·eq(t,i)]·C(fᵢ)` 中相同点的项在 MSM 中自然合并。

**涉及文件**:
- `subroutines/src/pcs/deMultilinear_kzg/batching.rs`
- 调用方构造 `points` 向量的代码

---

## 8. 遗留验证器路径清理

**来源**: NOTES #5

**现状**: `verify_proof_eval()` 中 legacy 路径从原始 witness/selector 重建 MLE 并本地求值，与 PCS 验证路径并存。

**优化**: 废弃或 gate behind feature flag；如保留则并行化内部求值。

**安全性论证**: legacy 路径与 PCS 路径验证相同的不变量 `c = v · eq(ρ, r_b)`。移除不影响协议正确性，只影响验证的实现方式。

**涉及文件**:
- `deSnark/src/snark.rs` — `verify_proof_eval()`
