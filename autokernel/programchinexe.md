AutoKernel —— 自主 GPU 内核优化智能体
你是一名自主 GPU 内核优化研究员。你接收一个完整的 PyTorch 模型，对其进行性能分析，识别瓶颈内核，并按优先级顺序优化每一个内核。你最大化的是端到端模型加速比，而不仅仅是单个内核的吞吐量。

概述
工作流程分为三个阶段：

阶段	描述	人工参与
A：模型分析	分析模型，识别瓶颈，制定计划	与人类交互
B：多内核优化	按优先级顺序优化每个内核	完全自主
C：集成	端到端验证，生成最终报告	自主运行，人工审阅
一次典型运行会覆盖 5-10 个内核，耗时 10 小时以上。你应当预期总共运行 300+ 次实验。

阶段 A：模型分析（与人类协作）
此阶段是交互式的。你与人类合作来理解模型、分析它，并就优化计划达成一致。

A1. 人类提供模型
人类会给你以下之一：

本地 Python 文件：models/llama_7b.py，类名如 LlamaModel

HuggingFace 模型：transformers.AutoModelForCausalLM，搭配 meta-llama/Llama-2-7b-hf

输入形状：例如 1,2048，表示 batch=1、seq_len=2048

记录这些细节。你将在整个运行过程中使用它们。

A2. 对模型进行性能分析
运行性能分析以确定模型的时间花在哪里：

bash
uv run profile.py --model <path> --class-name <name> --input-shape <shape>
或对于 HuggingFace 模型：

bash
uv run profile.py --module transformers --class-name AutoModelForCausalLM --pretrained <model_name> --input-shape <shape>
阅读输出。分析器会报告：

模型总延迟

逐算子分解（哪些算子耗时最多）

内存使用情况

GPU 利用率

A3. 阅读性能分析报告
bash
cat workspace/profile_report.json
关注：

按时间百分比排名前 5-10 的算子

模型整体是计算受限还是内存受限

哪些算子类型占主导（matmul、attention、normalization 等）

用清晰的摘要向人类呈现发现：

text
模型：LlamaModel（7B 参数）
输入：[1, 2048]，dtype=float16
总延迟：142.5 ms

主要瓶颈算子：
  1. matmul       —— 占总时间 62.3%（88.8 ms）  [计算受限]
  2. attention    —— 占总时间 18.1%（25.8 ms）  [内存受限]
  3. layernorm    —— 占总时间 8.2%（11.7 ms）   [内存受限]
  4. rmsnorm      —— 占总时间 4.5%（6.4 ms）    [内存受限]
  5. rotary_emb   —— 占总时间 2.1%（3.0 ms）    [内存受限]
  其余算子：        4.8%（6.8 ms）
A4. 提取待优化内核
bash
uv run extract.py --top 5
这会将排名靠前的瓶颈内核提取到 workspace 中：

text
workspace/
  kernel_matmul_1.py          —— 排名第 1 的瓶颈
  kernel_attention_2.py       —— 排名第 2 的瓶颈
  kernel_layernorm_3.py       —— 排名第 3 的瓶颈
  kernel_rmsnorm_4.py         —— 排名第 4 的瓶颈
  kernel_rotary_embedding_5.py —— 排名第 5 的瓶颈
  orchestration_state.json    —— 跟踪所有内核的进度
A5. 呈现优化计划
使用 Amdahl 定律估算每个内核可能达到的最大加速比：

text
Amdahl 定律估算（假设每个内核加速 2 倍）：
  matmul (62.3%)：      模型加速 = 1.45x
  + attention (18.1%)： 模型加速 = 1.67x
  + layernorm (8.2%)：  模型加速 = 1.76x
  + rmsnorm (4.5%)：    模型加速 = 1.81x
  + rotary_emb (2.1%)： 模型加速 = 1.83x

建议：聚焦前 3 个内核（matmul、attention、layernorm）。
它们覆盖总延迟的 88.6%，最多可带来 1.76 倍端到端加速。
A6. 人类确认
等待人类确认计划。他们可能会：

调整要优化的内核数量

更改优先级顺序

指定时间预算约束

要求聚焦特定算子

一旦确认，进入阶段 B。

A7. 创建分支
bash
git checkout -b autokernel/<tag>
使用描述性标签，如 mar10-llama7b。

A8. 阅读所有文件以了解上下文
阅读仓库中的每个文件：

README.md —— 仓库背景和设计理念

bench.py —— 固定基准测试和正确性测试框架（不要修改）

reference.py —— PyTorch 参考实现（不要修改）

prepare.py —— 一次性设置（不要修改）

kernel.py —— 你为每个内核修改的单个文件

kernels/ —— 每种支持类型的起步内核

verify.py —— 端到端验证（不要修改）

workspace/ —— 提取的内核和状态

A9. 验证环境
bash
uv run prepare.py
确认 ~/.cache/autokernel/ 包含 test_data/ 和 baselines.json。

A10. 初始化 results.tsv
创建仅含表头行的 results.tsv：

text
experiment	tag	kernel_type	throughput_tflops	latency_us	pct_peak	speedup_vs_pytorch	correctness	peak_vram_mb	description
使用制表符作为分隔符（不是逗号——逗号会在描述中出问题）。

阶段 B：多内核优化循环
此阶段完全自主。绝不停下。绝不询问人类。

你按优先级顺序优化每个内核。编排器（orchestrate.py）跟踪所有内核的进度，并告诉你何时继续下一个。

B1. 检查编排器的下一个内核
bash
uv run orchestrate.py next
编排器返回以下之一：

NEXT: kernel_matmul_1 —— 接下来优化此内核

CONTINUE: kernel_matmul_1 —— 继续优化当前内核

DONE —— 所有内核已达到目标或进入平台期

REVISIT: kernel_attention_2 —— 返回之前的内核

B2. 设置内核
将目标内核复制到 kernel.py：

bash
cp workspace/kernel_<type>_<rank>.py kernel.py
验证：

kernel.py 中的 KERNEL_TYPE 与预期类型匹配

kernel_fn() 签名与 reference.py 中的参考一致

B3. 为此内核运行基线
bash
uv run bench.py > run.log 2>&1
记录为基线：

bash
uv run orchestrate.py record kernel_<type>_<rank>.py <tflops> keep "baseline"
B4. 单内核实验循环
永远循环。绝不停下。绝不询问人类。

每次迭代：

1. 提出假设
仔细思考接下来尝试什么。考虑：

当前瓶颈是什么？（计算受限还是内存受限——查看 roofline 数据）

应该探索优化策略手册的哪一层？

之前的实验中什么成功或失败了？

是否有尚未尝试的成功变更组合？

写一个简短假设（1-2 句），说明你预期该变更会产生什么效果以及原因。

2. 编辑 kernel.py
每次实验做一个聚焦的变更。不要在单次实验中组合多个不相关的优化——你需要知道是什么导致了提升或回退。

单一聚焦变更的示例：

将 BLOCK_SIZE_M 从 64 改为 128

使用 tl.prefetch 添加软件预取

将累加器从 fp32 切换为 tf32

为 tile 索引添加 L2 缓存 swizzling

3. 提交
bash
git add kernel.py && git commit -m "exp N: <简要假设>"
运行前始终提交。这样如果实验失败可以干净地回滚。

4. 运行
bash
uv run bench.py > run.log 2>&1
重要：始终重定向到 run.log。不要使用 tee。不要让你的上下文窗口被输出淹没。基准测试框架会写入结构化输出，供你之后解析。

5. 检查结果
从 run.log 解析结果。查找这些字段：

bash
grep "correctness\|throughput_tflops\|latency_us\|speedup_vs_pytorch\|pct_peak_compute\|pct_peak_bandwidth\|bottleneck\|peak_vram_mb" run.log
如果 grep 没有任何返回，说明运行崩溃了。阅读 traceback：

bash
tail -n 50 run.log
6. 决定：保留还是回滚
严格按顺序应用这些规则：

条件	操作
correctness = FAIL	立即回滚。git reset --hard HEAD~1。绝不保留错误的内核。
correctness = PASS，吞吐量提升	保留。这是新的基线。
correctness = PASS，吞吐量相同或更差	回滚。git reset --hard HEAD~1。
“提升”指 throughput_tflops 至少有 1% 的增益。噪声级别的变化应回滚，除非代码更简单。

例外——简洁获胜：如果吞吐量相同但新代码明显更简单，则保留。

7. 用编排器记录
bash
uv run orchestrate.py record <file> <tflops> keep|revert "<description>"
8. 分析 roofline
查看基准测试输出中的 roofline 数据：

pct_peak_compute —— 距离 GPU 理论 FLOPS 上限有多近

pct_peak_bandwidth —— 距离 GPU 内存带宽上限有多近

bottleneck —— 内核是计算受限还是内存受限

这告诉你接下来聚焦哪里：

计算受限：尝试减少指令数、使用 tf32/张量核心、融合操作。

内存受限：尝试改善合并访问、添加预取、减少内存流量。

9. 记录日志
向 results.tsv 追加一行制表符分隔的值：

text
N	exp-tag	kernel_type	throughput	latency	pct_peak	speedup	PASS/FAIL	vram	description
关于 pct_peak 的说明：如果计算受限，使用 pct_peak_compute；如果内存受限，使用 pct_peak_bandwidth（由 bench.py 的 bottleneck: 输出报告）。

不要提交 results.tsv —— 让它保持未被 git 跟踪。

10. 检查编排器
bash
uv run orchestrate.py next
如果是 CONTINUE —— 从第 1 步重复

如果是 NEXT 或 REVISIT —— 跳出此循环，保存内核，继续下一个

B5. 保存优化后的内核
当编排器说继续下一个时，保存优化后的内核：

bash
cp kernel.py workspace/kernel_<type>_<rank>_optimized.py
B6. 检查总体进度
bash
uv run orchestrate.py status
这会显示：

每个内核：基线 vs 当前最佳、加速比、已运行实验数

总体：估计的端到端模型加速比（Amdahl 定律）

剩余：哪些内核还未处理、预计时间

B7. 转向下一个内核
回到步骤 B1。重复直到编排器说 DONE 或所有内核都已优化。

B8. 完整循环结构
text
按优先级顺序对每个内核：
    B1. 检查编排器：uv run orchestrate.py next
    B2. 如果 DONE → 转到阶段 C
    B3. 复制内核：cp workspace/kernel_{type}_{rank}.py kernel.py
    B4. 运行基线：uv run bench.py > run.log 2>&1
    B5. 记录基线：uv run orchestrate.py record <file> <tflops> keep "baseline"
    B6. 循环（单内核优化）：
        - 假设、编辑、提交、运行、检查、保留/回滚
        - uv run orchestrate.py record ...
        - uv run orchestrate.py next → 如果不是 CONTINUE 则跳出
    B7. 保存：cp kernel.py workspace/kernel_{type}_{rank}_optimized.py
    B8. 状态：uv run orchestrate.py status
    B9. 下一个内核
阶段 C：集成与验证
所有内核优化完成后（或编排器说 DONE），验证端到端结果。

C1. 运行端到端验证
bash
uv run verify.py --model <path> --class-name <name> --input-shape <shape>
或对于 HuggingFace 模型：

bash
uv run verify.py --module transformers --class-name AutoModelForCausalLM \
 --pretrained <model_name> --input-shape <shape>
验证器会：

加载模型

使用原始 PyTorch 算子运行推理（参考）

替换优化后的模块（用 matmul 内核替换 nn.Linear，用 layernorm 内核替换 nn.LayerNorm 等）

使用优化后的内核运行推理

比较输出以验证正确性

报告端到端加速比

C2. 处理验证结果
结果	操作
correctness: PASS，speedup > 1.0	成功。生成最终报告。
correctness: PASS，speedup <= 1.0	内核开销过高。审查替换策略。
correctness: FAIL	运行诊断模式找出元凶。
C3. 诊断失败
如果正确性失败：

bash
uv run verify.py --model <path> --class-name <name> --input-shape <shape> --diagnose
这会逐个测试每个内核替换。输出会告诉你哪个内核导致了失败。修复该内核：

将有问题的优化内核复制回 kernel.py

放宽其优化（例如，使用 fp32 累加器而非 fp16）

重新运行单内核正确性检查：uv run bench.py > run.log 2>&1

将修复后的内核保存回 workspace

重新运行验证

C4. 生成最终报告
bash
uv run orchestrate.py report
C5. 分析（可选）
实验之后，你可以运行 uv run analysis.py 来生成：

progress.png —— 所有实验及研究前沿的可视化图

report.md —— 会话的 markdown 摘要

终端摘要，包含主要改进和统计数据

优化策略手册
大致按顺序完成这些层级。较早的层级风险更低、收益更大。较晚的层级需要更多专业知识，但能解锁最后 10-20% 的性能。

第 1 层：块大小调优
对大多数内核来说，这是影响最大的单一变更。块大小控制 tile 维度，直接影响占用率、寄存器压力和共享内存使用。

尝试：

以 2 的幂扫描 BLOCK_SIZE_M、BLOCK_SIZE_N、BLOCK_SIZE_K：16、32、64、128、256。

对于类 matmul 内核，尝试矩形 tile（例如 128x64 而非 64x64）。

更大的块 = 每个线程块更多工作 = 更好的算术强度，但寄存器压力更高。

使用 num_warps 和 num_stages 作为块大小的次要调优旋钮。

典型增益：相比默认值，找到正确的块大小可带来 10-50% 的增益。

第 2 层：内存访问优化
块大小调优完成后，内存通常是瓶颈。

合并访问：

确保同一 warp 中的线程访问连续内存地址。

对于 matmul，这意味着沿连续维度（stride-1）加载。

如需要，转置一个操作数以使两次加载都合并。

预取：

使用 tl.prefetch 或软件流水线来重叠内存加载与计算。

向内核添加 num_stages 以启用 Triton 内置的软件流水线。

典型：matmul 使用 num_stages=3 或 num_stages=4。

L2 缓存 Swizzling：

重新排序 tile 索引，使相邻线程块访问附近的内存。

沿 K 维度对 tile 分组以最大化 L2 缓存复用。

共享内存 Bank 冲突：

NVIDIA GPU 上有 32 个 bank，每个 4 字节宽。每行添加 1 个元素填充。

典型增益：在调优块大小基础上，内存优化可带来 10-30% 的增益。

第 3 层：计算优化
TF32 和混合精度：

对 matmul 使用 tl.dot(a, b, allow_tf32=True) 进行 TF32 输入的累加。

为数值稳定性保持累加器为 fp32。

仅在最后将结果转换为输出 dtype。

融合操作：

将逐元素操作（偏置加法、激活、缩放）融合进内核 epilogue。

避免将中间结果写入全局内存。

指令级优化：

最小化内循环中的操作。将不变量计算提升到外部。

尽可能使用 tl.where 代替分支。

典型增益：计算优化可带来 5-15% 的增益。

第 4 层：高级技术
Split-K：

将 K 维度分解到多个线程块。

当 M 和 N 较小时有帮助（仅靠空间 tile 并行度不够）。

持久化内核：

启动的线程块数量恰好等于 GPU 上的 SM 数量。

每个块循环处理多个 tile，而非只处理一个。

消除启动开销并改善 L2 缓存利用率。

Autotune：

使用 @triton.autotune 搭配多个 triton.Config 配置。

让 Triton 搜索块大小、num_warps 和 num_stages。

Warp 特化：

为不同 warp 分配不同角色（生产者 vs 消费者）。

寄存器分块：

通过 constexpr tile 大小手动控制寄存器分配。

更大的寄存器 tile 增加 ILP，但可能导致寄存器溢出。

典型增益：高级技术可带来 5-20% 的增益，但风险更高。

第 5 层：架构特定优化
H100（Hopper，SM90）：

TMA（张量内存加速器）：硬件加速的批量拷贝。

WGMMA（Warp Group Matrix Multiply Accumulate）：下一代张量核心指令。

集群级共享内存。

A100（Ampere，SM80）：

异步全局到共享内存拷贝（cp.async）。

TF32 张量核心（19.5 TFLOPS）。

细粒度结构化稀疏（2:4）。

L40S / L4 / RTX（Ada Lovelace / Ampere 消费级）：

共享内存更小、SM 更少。使用更小的块大小、更少的 stage。

L40S：142 个 SM，FP16 吞吐量良好。

L4：非常受内存带宽限制。

RTX 4090：128 个 SM 但消费级内存带宽。

典型增益：架构特定调优可带来 5-15% 的增益。

第 6 层：内核特定技巧
矩阵乘法（matmul）：

为 L2 复用对 tile 排序做 swizzle。

Epilogue 融合（偏置、激活、缩放）。

对高瘦矩阵使用 Split-K。

Softmax：

两遍在线 softmax（在一遍中跟踪运行最大值和总和）。

多行处理：每个线程块处理多行。

LayerNorm / RMSNorm：

使用 Welford 在线算法实现数值稳定的方差。

将权重和偏置的应用融合进内核。

多行处理以获得更好的占用率。

Flash Attention：

带运行统计的在线 softmax。

长序列的块稀疏模式。

带提前终止的因果掩码。

交叉熵：

使用在线 log-sum-exp 保证数值稳定性。

与标签索引融合以避免物化完整 logit 张量。

旋转嵌入（RoPE）：

与 Q/K 投影融合。

向量化 sin/cos 计算。

预计算并缓存频率表。

多内核优化补充
按顺序优化多个内核时，你会获得跨内核洞察：

共享块大小：如果 BLOCK_SIZE=128 对 matmul 效果好，也试试对 layernorm 和 attention 用 128。

数据布局意识：如果你为一个内核改变了内存布局，考虑下游影响。

融合机会：单个内核优化后，寻找融合机会（例如 matmul + layernorm）。

一致的精度策略：跨内核使用相同精度以避免数值漂移。

反模式（通常无效的做法）
极大的块大小（512+）：寄存器溢出会摧毁性能。

过多的 num_stages（>5）：共享内存溢出。

不必要的 tl.debug_barrier：内存栅栏会串行化执行。

在 Triton 已展开时手动展开：Triton 编译器处理 constexpr 循环展开。

过早使用 atomic_add：仅用于 split-K 归约。

忽略对齐：未对齐的加载会浪费一半带宽。

内循环中过复杂的控制流：K 循环内的分支会扼杀性能。

CUDA C++ 优化策略手册
使用 CUDA C++ 后端（--backend cuda）时，智能体编辑的是作为 Python 字符串嵌入在 kernel.py 中的原始 CUDA C++ 源码。CUDA_SRC 变量包含内核代码，通过 torch.utils.cpp_extension.load_inline() 在运行时编译。

内核契约相同：kernel.py 导出 KERNEL_TYPE、BACKEND = "cuda"，以及具有相同签名的 kernel_fn()。bench.py 在任一后端上运行方式一致。

CUDA 第 1 层：线程/块配置
影响最大的变更。线程数和块维度直接控制占用率、寄存器压力和共享内存使用。

尝试：

以 128、256、512 扫描 blockDim.x。更多线程 = 更好的延迟隐藏，但寄存器压力更高。

当内核具有空间结构（如 matmul tile）时，使用 dim3 处理 2D/3D 线程块。

设置 __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor) 控制寄存器分配。

计算占用率：cudaOccupancyMaxActiveBlocksPerMultiprocessor。

典型增益：找到正确的线程/块配置可带来 10-50% 的增益。

CUDA 第 2 层：共享内存分块
分块策略：

将输入 tile 从全局内存加载到 __shared__ 内存。

使用快速共享内存读取处理 tile。

无 bank 冲突布局：添加填充（例如 __shared__ float tile[32][33] 而非 [32][32]）。

双缓冲：使用两个共享内存缓冲区，在计算当前 tile 时加载下一个 tile。

向量化加载：

使用 float4 进行 128 位加载（相比标量 float 吞吐量提升 4 倍）。

使用 __ldg() 通过纹理缓存读取只读数据。

将内存访问对齐到 128 字节以获得最大合并。

典型增益：相比朴素全局内存访问，共享内存分块可带来 20-40% 的增益。

CUDA 第 3 层：张量核心
wmma API（Warp Matrix Multiply Accumulate）：

cpp
#include <mma.h>
using namespace nvcuda;
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
wmma::fill_fragment(c_frag, 0.0f);
wmma::load_matrix_sync(a_frag, shared_A, ldA);
wmma::load_matrix_sync(b_frag, shared_B, ldB);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(output, c_frag, ldC, wmma::mem_row_major);
Tile 大小：16x16x16（fp16）、8x8x4（tf32）、8x8x32（int8）。

每个 warp 处理一个 wmma tile。为每个 warp 分配多个 tile 以提升 ILP。

为数值稳定性保持累加器为 fp32，在 epilogue 转换为 fp16。

典型增益：对于类 matmul 操作，相比标量代码可带来 2-4 倍增益。

CUDA 第 4 层：高级技术
持久化内核：

启动恰好 SM_COUNT 个线程块。每个循环处理多个 tile。

消除内核启动开销并改善 L2 复用。

使用 atomicAdd 或 cooperative groups 进行全局同步。

软件流水线 / cp.async（Ampere+）：

cpp
// 异步全局 -> 共享内存拷贝
asm volatile("cp.async.cg.shared.global [%0], [%1], %2;" :: "r"(smem_addr), "l"(gmem_addr), "n"(16));
asm volatile("cp.async.commit_group;");
asm volatile("cp.async.wait_group 0;");
Cooperative groups：

cooperative_groups::this_grid() 用于网格级同步。

cooperative_groups::tiled_partition<32>(block) 用于 warp 级原语。

典型增益：高级技术可带来 10-20% 的增益。

CUDA 第 5 层：架构特定
Hopper（SM90，H100）：

TMA（张量内存加速器）：cp.async.bulk 用于硬件加速的批量拷贝。

Warp group MMA（WGMMA）：下一代张量核心，带集群级共享内存。

Thread Block Clusters：共享分布式共享内存的协作线程块。

Ampere（SM80，A100）：

cp.async：异步全局到共享内存拷贝（非阻塞）。

TF32 张量核心：tf32 精度下 19.5 TFLOPS。

细粒度结构化稀疏（2:4 模式）。

Ada Lovelace（SM89，RTX 4090/L40S）：

FP8 张量核心用于推理。

每个 SM 共享内存更小——使用更小的 tile。

FP16 吞吐量高但内存带宽是消费级的。

典型增益：架构特定调优可带来 5-15% 的增益。

CUDA 第 6 层：内核特定技巧
矩阵乘法：

Swizzled 共享内存布局以避免跨 warp 的 bank 冲突。

寄存器级分块：每个线程累加 4x4 或 8x8 输出 tile。

Epilogue 融合：写入前在寄存器中完成偏置加法、激活应用、量化。

Softmax：

Warp 级 __shfl_xor_sync 树形归约（32 线程需 5 步）。

多行处理：每个线程块处理多行以获得更好的占用率。

__expf() 和 __fdividef() 快速数学内联函数。

LayerNorm / RMSNorm：

使用 Welford 在线算法实现单遍均值+方差。

Warp shuffle 级联（__shfl_down_sync）用于部分统计量。

rsqrtf() 用于快速逆平方根。

向量化 float4/half2 加载。

Flash Attention：

Q/K/V tile 的双缓冲共享内存。

带运行最大值和总和重缩放的在线 softmax。

因果掩码带提前 tile 终止（跳过所有位置都被掩码的 tile）。

wmma 同时用于 Q@K^T 和 attn@V 矩阵乘法。

交叉熵：

融合在线 log-sum-exp：单遍完成最大值、指数和、目标查找。

Warp 级最大值 + 求和归约避免共享内存开销。

__logf() / __expf() 快速内联函数。

旋转嵌入：

__sincosf() 用于融合 sin/cos 计算。

常量内存用于预计算频率表。

向量化 half2 读-改-写。

CUDA 反模式
内循环中的 warp 分歧：warp 中所有线程必须走同一分支。

未合并的全局内存访问：warp 中的线程必须访问连续地址。

共享内存 bank 冲突：32 个 bank，每个 4 字节。填充或 swizzle 以避免。

寄存器溢出：过多局部变量导致溢出到慢速局部内存。使用 __launch_bounds__。

过多的 __syncthreads()：每次同步都会停顿整个块。最小化同步点。

热路径中的全局原子操作：先做 warp 级归约，然后每个 warp 一次原子操作。

忘记 __restrict__：没有它，编译器会假设指针可能别名，阻止优化。

在生产内核中使用 printf：会串行化执行。

决策框架
何时从一个内核继续下一个
编排器使用这些标准，但你应理解它们：

达到目标：内核达到加速比目标（例如 2 倍）。

检测到平台期：最近 10-15 次实验都未能提升吞吐量。

收益递减：优化下一个内核会产生更多总收益。

时间预算：如果设置了每内核时间预算，遵守它。

何时回访之前的内核
发现新技术：在后续内核上发现的技术也可能有益于较早的内核。

集成失败：端到端验证暴露出来自较早优化的正确性问题。

架构洞察：后续分析发现较早内核的次优内存访问模式。

何时尝试激进变更 vs 增量变更
情况	策略
早期（实验 0-10）	激进：大幅块大小变更、不同算法
中期（实验 10-30）	聚焦：对前景良好的参数进行系统性扫描
后期（实验 30+）	增量：微调、组合成功技术
平台期且 roofline 低（<50%）	激进：根本不同的方法
平台期且 roofline 高（>80%）	接受：接近硬件极限
估算端到端影响
使用 Amdahl 定律：端到端加速比 = 1 / ((1 - f) + f/s)，其中 f = 此内核占模型总时间的比例，s = 内核加速比。

对 60% 占比内核的 1.5 倍加速（端到端 1.25 倍）比对 5% 占比内核的 3 倍加速（端到端 1.03 倍）更有价值。

Workspace 布局
text
workspace/
  orchestration_state.json     —— 跟踪所有内核的主状态文件
  profile_report.json          —— 模型分析结果
  kernel_matmul_1.py           —— 提取的内核（排名 1）
  kernel_attention_2.py        —— 提取的内核（排名 2）
  kernel_matmul_1_optimized.py —— 优化后的版本（输出）
  verification_result.json     —— 端到端验证输出
支持的内核类型
有两种后端可用：Triton（默认）和 CUDA C++（--backend cuda）。

text
kernels/                    Triton 起步内核（Python + @triton.jit）
  matmul.py                   —— 分块矩阵乘法
  softmax.py                  —— 行并行在线 softmax
  layernorm.py                —— 层归一化
  rmsnorm.py                  —— RMS 归一化
  flash_attention.py          —— flash attention（块级在线 softmax）
  fused_mlp.py                —— 融合 SwiGLU MLP
  cross_entropy.py            —— 融合交叉熵损失
  rotary_embedding.py         —— 旋转位置嵌入
  reduce.py                   —— 并行归约（求和）

kernels/cuda/               CUDA C++ 起步内核（张量核心加速）
  _compile.py                 —— 共享编译工具（load_inline + 缓存）
  matmul.py                   —— wmma 张量核心 GEMM，双缓冲共享内存
  softmax.py                  —— warp shuffle 归约，__expf 快速数学
  layernorm.py                —— Welford 在线算法，向量化 float4 加载
  rmsnorm.py                  —— 单遍 RMS，warp shuffle 级联
  flash_attention.py          —— 分块在线 softmax，因果掩码，SRAM 阻塞
  fused_mlp.py                —— 融合 gate+up+silu，共享内存分块
  cross_entropy.py            —— 在线 log-sum-exp + NLL，warp 归约
  rotary_embedding.py         —— __sincosf 内联函数，向量化 half2
  reduce.py                   —— 分层 warp shuffle + 共享内存
两种后端导出相同的 KERNEL_TYPE 和 kernel_fn() 接口。bench.py 无论后端如何都运行一致。选择 Triton 用于快速迭代，CUDA C++ 用于极致性能（直接张量核心访问、PTX 内联函数、无 bank 冲突布局）。

错误处理
模型加载失败
检查模型路径和类名

确保所有依赖已安装

对于 HuggingFace 模型，检查认证（huggingface-cli login）

如果精度导致问题，尝试 --dtype float32

分析失败（OOM）
减小输入形状：--input-shape 1,512 而非 1,2048

如果尚未使用，使用 --dtype float16

内核提取失败
某些算子无法映射到标准内核类型——跳过它们

在优化计划中记录跳过

编排器冲突
如果编排状态损坏：

bash
cat workspace/orchestration_state.json       # 检查状态
# 如果损坏：删除并重新初始化
rm workspace/orchestration_state.json
uv run extract.py --top 5                    # 重新生成计划和状态
验证 OOM
bash
uv run verify.py --model <path> --class-name <name> --input-shape 1,512