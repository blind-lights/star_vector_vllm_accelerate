AutoKernel
https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white

面向 GPU 内核的自动研究。 给它任意一个 PyTorch 模型，你去睡觉，醒来时就能得到优化后的 Triton 或 CUDA C++ 内核。

https://progress.png/

受 @karpathy/autoresearch 启发——该项目展示了用于 LLM 训练研究的自主 AI 智能体。AutoKernel 将同样的理念应用于 GPU 内核优化：智能体修改一个文件，运行固定的评估，保留或回滚，如此循环往复。

工作原理
给 AutoKernel 任意一个 PyTorch 模型。它会：

分析（Profile） 模型，找出哪些 GPU 内核是瓶颈

提取（Extract） 每个瓶颈为独立的 Triton 或 CUDA C++ 内核

优化（Optimize） 每个内核，自主进行（编辑、基准测试、保留/回滚——永不停歇）

验证（Verify） 端到端正确性，并报告总加速比

智能体读取 program.md——即“研究组织代码”——其中包含自主运行的详尽指令。它一次编辑一个内核（kernel.py），运行 bench.py（固定基准测试，含 5 阶段正确性检查 + roofline 分析），然后保留或回滚更改。编排器使用 Amdahl 定律决定何时转向下一个内核。

每次实验约需 90 秒。即每小时约 40 次实验，一夜约 320 次，覆盖所有内核。

快速开始
环境要求： NVIDIA GPU（已在 H100/A100/RTX 4090 上测试）、Python 3.10+、uv。

bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆并安装
git clone https://github.com/RightNow-AI/autokernel.git
cd autokernel
uv sync

# 一次性设置：测试数据 + 基线
uv run prepare.py

# 分析模型（自带 GPT-2、LLaMA、BERT——无需 transformers）
uv run profile.py --model models/llama_7b.py --class-name LlamaModel \
 --input-shape 1,512 --dtype float16

# 提取排名靠前的瓶颈内核
uv run extract.py --top 5

# 验证基准测试可用
uv run bench.py
运行智能体
在本目录中启动 Claude、Codex 或任意编码智能体：

text
阅读 program.md，让我们启动一个新实验。从设置开始。
智能体将：

分析你的模型并给出优化方案

创建分支（例如 autokernel/mar10-llama7b）

按优先级顺序优化每个瓶颈内核

验证端到端正确性并报告总加速比

program.md 特意写得非常详尽，以便智能体能连续运行 10 小时以上而不卡住。它包含 6 层优化策略手册、决策框架、崩溃处理和 Amdahl 定律推理。

流水线
text
                 profile.py              extract.py           bench.py (循环)         verify.py
任意 PyTorch  ──>  按 GPU 时间    ──>  生成基线        ──>  优化每个        ──>  端到端
   模型           对内核排名          Triton/CUDA 内核      内核（智能体）      验证
工具	作用
profile.py	使用 torch.profiler 分析任意 PyTorch 模型，按 GPU 时间对内核排名，分类为计算/内存受限
extract.py	将排名前 N 的瓶颈内核提取为独立的 Triton 或 CUDA C++ 内核文件（--backend triton|cuda）
orchestrate.py	多内核调度器：使用 Amdahl 定律决定下一个优化哪个内核，跟踪总体进展
bench.py	固定基准测试：5 阶段正确性（冒烟、形状扫描、数值稳定性、确定性、边界情况）+ 性能 + roofline
verify.py	将优化后的内核插回模型，检查端到端正确性，报告总加速比
支持的内核
涵盖现代深度学习核心操作的 9 种内核类型：

内核	描述	关键指标
matmul	稠密矩阵乘法 (M x K) @ (K x N)	TFLOPS
softmax	行并行数值稳定 softmax	GB/s
layernorm	带仿射变换的层归一化	GB/s
rmsnorm	RMS 归一化（LLaMA 风格）	GB/s
flash_attention	带因果掩码的缩放点积注意力	TFLOPS
fused_mlp	SwiGLU 风格融合 MLP（gate + up + down）	TFLOPS
cross_entropy	融合交叉熵损失	GB/s
rotary_embedding	旋转位置嵌入（RoPE）	GB/s
reduce	并行归约（求和）	GB/s
每种内核在 reference.py 中都有 PyTorch 参考实现，在 kernels/ 中有起步 Triton 内核，在 kernels/cuda/ 中有起步 CUDA C++ 内核。

示例模型
AutoKernel 自带自包含的模型定义（无需 transformers 库）：

模型	文件	参数量	用法
GPT-2 Small	models/gpt2.py	124M	--class-name GPT2 --input-shape 1,1024
LLaMA（紧凑版）	models/llama_7b.py	160M	--class-name LlamaModel --input-shape 1,512
LLaMA 7B	models/llama_7b.py	7B	--class-name LlamaModel7B --input-shape 1,2048
BERT-base	models/bert_base.py	110M	--class-name BertModel --input-shape 8,512
自定义	models/custom.py	--	你自己的模型模板
对于 HuggingFace 模型（uv sync --extra models）：

bash
uv run profile.py --module transformers --class-name AutoModelForCausalLM \
 --pretrained meta-llama/Llama-2-7b-hf --input-shape 1,2048 --dtype float16
KernelBench 集成
AutoKernel 集成了 KernelBench——评估 AI 生成 GPU 内核的标准基准（跨 4 个难度级别的 250+ 道题）。大多数 KernelBench 评估使用一次性 LLM 生成，而 AutoKernel 对每道题运行 50-300+ 次迭代优化实验——系统性地探索优化空间，而非猜测。

bash
# 安装 KernelBench 依赖
uv sync --extra kernelbench

# 从 HuggingFace 获取 Level 1 题目
uv run kernelbench/bridge.py fetch --source hf --level 1

# 为特定题目设置优化环境
uv run kernelbench/bridge.py setup --level 1 --problem 1 --source hf

# 评估（正确性 + 相对 PyTorch 参考的加速比）
uv run kernelbench/bench_kb.py

# 对整级批量评分（计算 fast_p 指标）
uv run kernelbench/scorer.py --level 1
智能体读取 kernelbench/program_kb.md 获取 KernelBench 专属优化指令：如何编写 ModelNew 类、何时使用 CUDA C++ 还是 Triton、各题目级别的融合策略，以及适配 KernelBench fast_p 指标的编辑-基准-保留/回滚循环。

工具	作用
kernelbench/bridge.py	从 HuggingFace 或本地仓库加载题目，缓存，生成起步 kernel.py
kernelbench/bench_kb.py	评估 ModelNew vs Model：5 次试验正确性 + CUDA 事件计时 + 稳定性 + 确定性
kernelbench/scorer.py	对整级批量评估，计算各阈值（1.0x、1.5x、2.0x、3.0x、5.0x）下的 fast_p
kernelbench/program_kb.md	KernelBench 模式的智能体指令
HuggingFace Kernels 导出
将优化后的内核导出到 HuggingFace Hub 以便分发。用户随后只需一行即可加载你的内核：

python
from kernels import get_kernel
module = get_kernel("your-username/kernel-name")
bash
# 导出优化后的 CUDA 内核
uv run export_hf.py --name my_matmul

# 上传到 Hub（需要 `pip install kernels` 和 `huggingface-cli login`）
cd workspace/hf_export/my_matmul
kernels upload . --repo_id your-username/my_matmul
项目结构
text
autokernel/
  kernel.py             智能体修改的文件（一次一个内核）
  program.md            智能体指令——“研究组织代码”

  bench.py              固定基准测试 + 5 阶段正确性测试框架
  reference.py          PyTorch 参考实现（基准真值）
  prepare.py            一次性设置：测试数据、基线

  profile.py            分析任意 PyTorch 模型，按 GPU 时间对内核排名
  extract.py            将瓶颈内核提取到 workspace/
  orchestrate.py        多内核调度器（Amdahl 定律）
  verify.py             端到端模型验证 + 加速比报告
  export_hf.py          将优化内核导出为 HuggingFace Kernels 格式
  analysis.py           实验可视化（生成 progress.png）

  kernels/              起步 Triton 内核（9 种）
  kernels/cuda/         起步 CUDA C++ 内核（9 种，张量核心加速）
  kernelbench/          KernelBench 集成（桥接、评估框架、评分器）
  models/               自包含模型定义（GPT-2、LLaMA、BERT）
  workspace/            运行时产物（已 gitignore）
设计选择
双后端：Triton + CUDA C++。 Triton 用于快速迭代（类似 Python 的语法，几秒内编译）。CUDA C++ 用于极致性能（通过 wmma 直接访问张量核心、PTX 内联函数、无共享内存 bank 冲突的布局）。Triton 通常能达到 cuBLAS 的 80-95%；CUDA C++ 可以匹敌或超越它。两种后端共享相同的 kernel_fn() 接口——bench.py 在任一后端上运行方式完全一致。

正确性优先。 基准测试在测量性能前会先对照 PyTorch 检查内核输出。快速但错误的内核会立即被回滚。这防止智能体通过产生垃圾来“优化”。

Amdahl 定律编排。 编排器按影响排序。对 60% 占比内核的 1.5 倍加速（端到端 1.25 倍）胜过对 5% 占比内核的 3 倍加速（端到端 1.03 倍）。当收益递减时它会继续前进。

只修改单个文件。 智能体只触碰 kernel.py。范围可控，差异可审查，回滚干净。

TSV 日志。 结果写入纯文本 results.tsv 文件。人类可读、对 git 友好、极易解析、无需基础设施。

结果格式
每次实验都记录到 results.tsv（制表符分隔）：

列	描述
experiment	顺序实验编号（0 = 基线）
tag	短标识符
kernel_type	哪个内核（例如 matmul）
throughput_tflops	测量吞吐量（越高越好）
latency_us	执行时间（微秒）
pct_peak	GPU 理论峰值的百分比
speedup_vs_pytorch	相对 PyTorch/cuBLAS 的加速比
correctness	PASS、FAIL、TIMEOUT 或 CRASH
peak_vram_mb	峰值 GPU 显存使用
description	尝试了什么
致谢
本项目是面向 GPU 内核的自动研究——直接受 Andrej Karpathy 的 autoresearch 启发，那是用于 LLM 训练的自主 AI 研究智能体的原始实验。Karpathy 展示了 AI 智能体可以整夜运行数百次实验，系统地探索搜索空间并记录每个结果。AutoKernel 将同样的循环——智能体编辑一个文件、运行固定评估、保留或回滚——应用于使用 Triton 和原生 CUDA C++ 的 GPU 内核优化领域。

KernelBench 集成基于斯坦福 Scaling Intelligence Lab 的 Simon Guo、Sean Resta 等人的工作。他们的论文 "KernelBench: Can LLMs Write GPU Kernels?"（2025）确立了评估 AI 生成 GPU 内核的标准基准。AutoKernel 通过应用迭代优化（每道题 300+ 次实验）而非一次性生成来扩展它。KernelBench 数据集和评估协议：ScalingIntelligence/KernelBench。

由 RightNow AI 构建。企业级 GPU 优化，请查看 RightNow Enterprise。

更新日志
v1.3.0
AMD ROCm GPU 支持：MI300X、MI325X、MI350X、MI355X 检测和规格（感谢 @andyluo7）

修复 Python 3.13+ 上 verify.py 的 SyntaxError

修复 CUDA flash_attention 忽略 sm_scale 参数的问题

修复 CUDA cross_entropy 返回错误 dtype 的问题

修复 Triton rotary_embedding 广播截断的问题

修复 Triton reduce 在非最后一维归约时的输出形状问题

v1.2.0
增强分析器：--export-trace、--memory-snapshot、--torch-compile-log 标志

通过 export_hf.py 导出 HuggingFace Kernels

v1.1.0
原生 CUDA C++ 后端，含 9 种起步内核（张量核心、warp 内联函数、共享内存分块）

KernelBench 集成（250+ 标准化 GPU 内核题目）

extract.py 的 --backend triton|cuda 标志

v1.0.0
初始发布：Triton 内核优化流水线，含 5 阶段正确性测试框架

完整细节见 CHANGELOG.md。

许可证
MIT