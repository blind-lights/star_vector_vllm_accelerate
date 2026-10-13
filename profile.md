# StarVector on vLLM — 交接说明

阶段时间、nsys kernel 时间和 TTFT 的 NCU 屋顶都已量完。下一步是 **改 `autokernel/`，用它的搜索循环去试预填 GEMM**（预填 MLP、预填其余、CLIP），目标是降低图像请求的 TTFT。不要重做环境，不要改回纯文本 benchmark，不要再做 LayerNorm / GELU 融合。算子时间以 nsys 和 NCU 为准，不要用按权重字节切出来的时间。

## 用户要求

- 目标是商用卡（RTX 4090 / 5060 一类）上的 **推理** 加速，算子要有自己的融合形状。项目要能写进秋招简历，必须有闭环：影响排序、瓶颈诊断、正确性、保留或回退。
- AutoKernel 在 `kernel-agent/autokernel`，独立 uv 环境，torch cu128。可以改它的评测形状和说明，用搜索循环试预填 GEMM。不要把它的 torch 装进 `agent_env`。要加速的仍是已经接好 StarVector 的这份本地 vLLM，不是 `starvector/vllm` 那个 fork。
- 优化对象是 **图像转 SVG**。纯文本 `/v1/completions` 测不到 CLIP，不能当作优化目标。
- 未测之前不要发明某张卡专用的 kernel 规格。Eager 上变快，不能直接说 CUDA graph 里也更快。decode 的 GEMV 不要默认 cuBLAS 已经最优。
- 不要 git commit，除非用户明确要求。优先函数式写法。不要写有 SQL 注入的代码。对用户用中文。

## 硬件和两个环境

- 机器：WSL2，用户 `blossonadmin`。GPU 是 **NVIDIA GeForce RTX 5060 Ti，sm_120**，显存约 **7.96 GiB**。`nvidia-smi` 不在默认 PATH 里，但 PyTorch 能看到这块卡。
- 服务环境：conda `agent_env`，Python 3.10，`/home/blossonadmin/miniconda3/envs/agent_env`。
- **当前 torch 是 2.7.1+cu128，torchvision 0.22.1+cu128。** 原来的 2.5.1+cu121 只有到 sm_90，会在 `nn.Embedding` 初始化时报 `no kernel image is available`。
- vLLM 源码：`/home/blossonadmin/kernel-agent/vllm`，版本 `0.1.dev4453+gf5b3f5a99.d20261011.cu133`。`requirements-cuda.txt` 和 `pyproject.toml` 仍写着 `torch==2.5.1`，那是过期约束。
- 系统 CUDA 是 **13.3**：`/usr/local/cuda-13.3`。`conda activate agent_env` 会覆盖 `CUDA_HOME`，日志里会出现 `overwriting variable ['CUDA_HOME']`。
- vLLM 的 `_C` 和 FlashAttention-2 已按 `TORCH_CUDA_ARCH_LIST=12.0` 重编，能在 5060 Ti 上加载。FlashAttention-3 是 Hopper 专用，**故意没编**。启动日志是 `Using Flash Attention backend`，并已捕获 CUDA graph（35 个 decode shape）。
- 链接时要能找到 WSL 的 `libcuda`：`/usr/lib/wsl/lib`。否则 `ld` 报 `cannot find -lcuda`。
- GCC 13 编 vLLM 时，修过 site-packages 里的 `torch/include/ATen/core/List_inl.h`（`operator[]` 改用 `std::ptrdiff_t`）。重装 torch 会丢掉这个补丁。
- `huggingface-hub` 必须保持 **0.36.2**。`datasets` 是 **3.6.0**。装 `datasets==5.1` 会把 hub 升到 2.x，当前 `transformers 4.51.3` 会坏。
- StarVector 源码只通过 `PYTHONPATH=/home/blossonadmin/workspace/star-vector` 引入。**不要** `pip install -e` 那个仓库：egg-info 属主是 root，而且它锁 `transformers==4.49`、`python>=3.11`。

## 权重（两个目录不是重复）

- StarVector：`/home/blossonadmin/kernel-agent/vllm/models/starvector-1b-im2svg`（约 4.8 GB）。`config.json` 的 `starcoder_model_name` 已改成本地路径，本机访问不了 `huggingface.co`。
- StarCoder 基座：`/home/blossonadmin/kernel-agent/vllm/models/starcoderbase-1b`（约 4.3 GB）。只给 1B 的分词器和结构配置用。LLM 权重在 StarVector 里，已经微调过。
- 模型目录里有复制来的 `starvector_arch.py`。1B 必须走 `SimpleStarVectorProcessor`。判断条件是路径以 `starcoderbase-1b` 结尾，不能再写死 Hub 名字 `bigcode/starcoderbase-1b`。走错会去找不存在的 `preprocessor_config.json`。

## 怎么启动

服务已经能起来。图像请求必须走 **chat**，并且带上 chat template，因为这个 tokenizer 没有自带模板。

```bash
export PYTHONPATH=/home/blossonadmin/workspace/star-vector:$PYTHONPATH
vllm serve /home/blossonadmin/kernel-agent/vllm/models/starvector-1b-im2svg \
  --trust-remote-code --dtype float16 --max-model-len 4096 \
  --gpu-memory-utilization 0.9 --port 8000
```

图像测试脚本：`vllm/benchmarks/test_benchmark_serving.py`。它把 `assets/examples` 下 11 张 PNG 发到 `/v1/chat/completions`。提示文本是 `<image-start><svg`。`<image-start>` 必须留在文本里，否则图像占位符会被模板丢掉，请求 500。请求体里还带了一段只拼接 text 的 `chat_template`。

```bash
conda activate agent_env
cd /home/blossonadmin/kernel-agent/vllm/benchmarks
python test_benchmark_serving.py \
  --backend openai-chat --host 127.0.0.1 --port 8000 \
  --endpoint /v1/chat/completions \
  --model /home/blossonadmin/kernel-agent/vllm/models/starvector-1b-im2svg \
  --tokenizer /home/blossonadmin/kernel-agent/vllm/models/starvector-1b-im2svg \
  --trust-remote-code \
  --dataset-name local-images \
  --num-prompts 11 \
  --max-concurrency 1 \
  --random-output-len 128 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99
```

图片目录：`/home/blossonadmin/workspace/star-vector/assets/examples/`（11 张，不含 checkpoint）。

## 当前指标（2026-10-11，单请求，5060 Ti）

优化基线用 **图像** 这组。每条含读图、CLIP、adapter，再生成 128 个 token。11/11 成功，墙钟 9.89 s。每条输入约 259 token，其中约 257 个是图像 token。

| 指标 | 均值 | 中位数 | P99 |
|---|---:|---:|---:|
| TTFT | 48.83 ms | 37.90 ms | 146.16 ms |
| TPOT | 6.69 ms | 6.68 ms | 6.74 ms |
| ITL | 6.64 ms | 6.64 ms | 7.58 ms |

输出吞吐 142.42 token/s，请求吞吐 1.11 req/s。P99 TTFT 146 ms 是首张图偏高，中位数是 38 ms。解码约 6.7 ms/token。

纯文本对照（`benchmark_serving.py`，`/v1/completions`，random，输入 1024，输出 128，20 条）**不是优化目标**：TTFT 均值 64.50 ms、中位数 64.60 ms、P99 66.17 ms，TPOT 6.67 ms，输出 140.33 token/s。它没有 CLIP。解码 TPOT 和图像路径几乎一样，只能当作 StarCoder 解码基线。

服务加载时：权重 2.69 GiB，KV cache 大约 4.14 GiB，CUDA blocks 22619。8 GB 卡上不要把 `max-model-len` 再加大。GPU 属性：36 个 SM，L2 32 MB。同一次 256 MB 的 device copy 测得 **386 GB/s**（读加写）。

## 阶段拆分（2026-10-11，CUDA Event，5060 Ti）

`torch.profiler` 的 CUDA 活动在这块卡上不可用：`CUPTI_ERROR_INVALID_DEVICE`，Chrome trace 里只有 CPU 事件。阶段时间用 CUDA Event，叶子模块用 forward hook。计时只在 `/start_profile` 到 `/stop_profile` 之间启用。CUDA graph 回放不会再进 Python hook，所以 **graph 模式下解码内部只有整段 replay 时间**；算子构成是 `--enforce-eager` 另跑的一次。

服务按上面的命令启动，另外要有：

```bash
export VLLM_TORCH_PROFILER_DIR=/home/blossonadmin/kernel-agent/vllm/benchmarks/profiles/starvector
export VLLM_PROFILER_WITH_STACK=0
```

客户端在图像 benchmark 上加 `--profile --profile-trace-dir $VLLM_TORCH_PROFILER_DIR`。结果写到该目录的 `phase_profile.json`。测量用的服务已经停掉。

### CUDA graph 开着（1 张图，输出 32 token）

TTFT 44.35 ms，TPOT **6.78 ms**。和上面 128 token 基线的 6.69 ms 一致，profiler 没有把解码打慢。

| 阶段 | 总 GPU 时间 | 次数 | 每次 |
|---|---:|---:|---:|
| decode（graph replay） | 181.25 ms | 32 | **5.66 ms** |
| sample（含词表投影） | 21.97 ms | 33 | **0.67 ms** |
| prefill（含视觉） | 28.92 ms | 1 | 28.92 ms |
| vision（CLIP + Adapter，嵌在 prefill 里） | 9.36 ms | 1 | 9.36 ms |

5.66 + 0.67 ≈ 6.3 ms，和 TPOT 6.78 ms 之间大约 0.4 ms 是 CPU 调度。预填的 LLM 部分约 28.9 − 9.4 ≈ 19.5 ms，其中 MLP 线性 12.29 ms、注意力 5.86 ms。Adapter 0.15 ms。

这次预填叶子时间：`prefill/llm/mlp_linear` 12.29 ms（48 次），`prefill/llm/attention` 5.86 ms（72 次），`vision/vision/mlp_linear` 4.75 ms（46 次）。graph 回放没有解码叶子时间。

### Eager 对照（`--enforce-eager`，1 张图，输出 16 token）

TTFT 40.84 ms，TPOT **9.93 ms**。解码 GPU 均值 **8.52 ms**，比 graph 的 5.66 ms 多出来的约 2.9 ms 是 kernel 启动，graph 已经拿掉了。不要用 Eager 成绩代替 serving。

解码每步（15 步合计除以 15）：

| 叶子 | 每步 | 15 步合计 |
|---|---:|---:|
| MLP 线性（`c_fc` + `c_proj`） | 4.40 ms | 65.95 ms |
| 注意力路径（投影 + Attention） | 2.51 ms | 37.60 ms |
| LayerNorm | 0.57 ms | 8.54 ms |
| GELU | 0.08 ms | 1.14 ms |
| 词表嵌入 / `lm_head` 调用 | 0.04 ms | 0.65 ms |

视觉：线性 4.88 ms，`nn.MultiheadAttention` 3.41 ms，norm 0.56 ms，conv 0.20 ms。Eager 的 sample 仍是 0.67 ms/步。

### 和权重大小对齐

StarCoder-1B：24 层，`n_embd=2048`，`n_inner=8192`，`n_head=16`，multi-query（KV 头 1 个，head_dim 128），词表 49152，fp16。每步权重大小：

- MLP：24 × 2 × (2048×8192×2 字节) ≈ **1.61 GB**。按 386 GB/s 是 4.17 ms，Eager 测到 4.40 ms。
- 注意力投影：24 × (2048×2304 + 2048×2048) × 2 字节 ≈ **0.43 GB**，约 1.1 ms。Eager 的 2.51 ms 还含 Attention kernel 和多次启动。
- 词表投影：49152×2048×2 ≈ **0.19 GB**，约 0.50 ms，对应 sample 的 0.67 ms。
- 解码开始时 KV 大约 3 MB，相对权重可以忽略。

上面这几毫秒是用权重字节除以 386 GB/s **估算**的，不是 graph 里面逐个 kernel 的时间。CUDA Event 包住的是整段 `cudaGraphLaunch`，拆不开内部算子。算子时间以下一节 nsys 为准。nsys 测到的 MLP GEMV 是 3.95 ms/步，注意力投影 GEMV 是 1.12 ms/步，和这里的 4.17 ms、1.11 ms 接近，但不要再引用估算值。

## nsys kernel 时间（2026-10-11，CUDA graph 按 node 展开）

Nsight Systems **2026.1.3**（`/usr/local/cuda-13.3/bin/nsys`）。`torch.profiler` 的 CUPTI 在 sm_120 上失败，nsys 的 CUDA 跟踪可以用。`--cuda-graph-trace=node` 会把 graph 回放拆成单个 kernel。采集范围是 `cudaProfilerApi`：服务先正常启动并捕获 graph，benchmark 的热身请求在 `cudaProfilerStart` 之前，随后只采集 1 张图、生成 32 个 token。

报告：`figures/nsys_starvector.nsys-rep`（旁边有 `.sqlite`）。柱状图：`figures/operator-time-share.png`。这张图替换了早先按字节比例画的那版。

这次采集的客户端 TPOT 是 **6.82 ms**，和 graph 基线 6.78 ms 一致。TTFT 被 node 跟踪抬到 299 ms，不要用这次的 TTFT。

解码 32 步的 kernel 时长加起来是 **177.4 ms，即 5.54 ms/步**。同一条路径上 CUDA Event 量到的整段回放是 5.66 ms。差出来的约 0.12 ms 是 kernel 之间的空隙，不是某个没记上的大算子。

grid 能对上层数：`(2048,16)` 和 `(512,32)` 各 768 次 = 24 层 × 32 步 × 1 个 MLP GEMV；`(576,16)` 和 `(512,16)` 各 768 次 = 每层两个注意力投影 GEMV；`(12304,16)` 共 33 次 = 词表 GEMV（49152 向上取整）。

| 算子 | 每步 | 32 步合计 |
|---|---:|---:|
| GEMV · MLP（`internal::gemvx`，每层 2 次） | **3.95 ms** | 126.31 ms |
| GEMV · 注意力投影（每层 2 次） | **1.12 ms** | 35.81 ms |
| FlashAttention 解码（`flash_fwd_splitkv` + `combine`，各 768 次） | **0.27 ms** | 8.70 ms |
| LayerNorm | 0.14 ms | 4.47 ms |
| GELU | 0.016 ms | 0.52 ms |
| 词表 GEMV（33 次，含预填那一次；每次 0.48 ms） | 0.48 ms | 15.71 ms |

预填和 CLIP 只跑一次，不在解码 graph 里。CUTLASS GEMM（`cutlass_80_tensorop_f16_s16816gemm_*`）合计 23.35 ms，其中 `64x256` 的 4.55 ms 落在 `vision` 范围里。FlashAttention 预填加 CLIP 的 `pytorch_flash::flash_fwd` 合计 1.23 ms。CLIP 的 `implicit_convolve_sgemm` 是 0.08 ms。softmax + argmax 是采样，33 次合计 0.94 ms。

这次采集里、按算子相加的 GPU kernel 时间大约 218 ms。占比：解码 MLP GEMV 58%，解码注意力投影 GEMV 16%，预填与 CLIP 的 GEMM 11%，词表 GEMV 7%，解码 FlashAttention 4%。生成 128 token 时，解码这两类 GEMV 的占比还会更高，预填和 CLIP 仍各只跑一次。

### CLIP 为什么不在这张 CUDA graph 里

graph 只在解码回放。`model_runner.py` 的 `execute_model` 在 `prefill_metadata is None` 且 `decode_meta.use_cuda_graph` 时才用 `graph_runners`。捕获输入是固定的 `input_ids`、`positions`、KV cache，没有 `pixel_values`。

`StarVectorModel.forward` 只在 `pixel_values` 存在时调用 `embed_image`。解码 128 步没有图像，捕获到的 kernel 序列里就没有 CLIP 的卷积、GEMM 和 attention。

单独为预填再捕获一张 graph，也只能钉死「224×224 + 某一个序列长度」。换图或序列长度变了，捕获时的地址和 launch 形状对不上。CLIP 一次大约 7–9 ms，时间主要在 GEMM 自己；graph 省的是启动。解码从 Eager 8.52 ms 到 graph 5.66 ms 的差距就是启动。CLIP 只跑一次，不值得为它再捕获 graph。

解码 GEMV 已经贴着显存带宽（MLP 3.95 ms/步，约 408 GB/s）。fp16 上再改 tile 空间很小。当前要动的是 TTFT 上的预填 GEMM。

## TTFT 与 TPOT 分开（nsys kernel 时间）

一次请求里的算子不要混在一张图里。图：`figures/operator-ttft.png`、`figures/operator-tpot.png`。旧的混合图 `figures/operator-time-share.png` 不要再当结论。

TTFT 是第一个 token 之前、只发生一次的 GPU kernel，合计 **26.22 ms**。CLIP 已从预填合计里拆出。

| 算子 | 时间 | 占这 26.22 ms |
|---|---:|---:|
| GEMM · 预填 MLP | 12.22 ms | 46.6% |
| GEMM · CLIP | 6.39 ms | 24.4% |
| GEMM · 预填其余 | 4.74 ms | 18.1% |
| FlashAttention · 预填 | 0.73 ms | 2.8% |
| FlashAttention · CLIP | 0.50 ms | 1.9% |
| GEMV · 词表（首 token） | 0.48 ms | 1.8% |

TPOT 是每个后续 token 的 GPU kernel，合计 **6.05 ms**。GEMV MLP 3.95 ms（65.3%），注意力投影 GEMV 1.12 ms（18.5%），词表 GEMV 0.48 ms（7.9%）。

层所在位置：

- 预填 MLP：`vllm/vllm/model_executor/models/gpt_bigcode.py` 的 `GPTBigMLP.c_fc` / `c_proj`。24 层各两次，对应 48 次较长的 `256x64` GEMM。
- 预填其余：同文件 `GPTBigCodeAttention` 的 `c_attn` / `c_proj`。仍是 `256x64`，每次更短。
- 这两处都经 `vllm/vllm/model_executor/layers/linear.py` 的 `UnquantizedLinearMethod.apply`，也就是 `F.linear`。fp16、无量化，不进 `csrc`。
- CLIP GEMM：`/home/blossonadmin/workspace/star-vector/starvector/model/image_encoder/clip_model.py` 的 `ResidualAttentionBlock`（`mlp.c_fc` / `c_proj`，以及 `MultiheadAttention` 的投影）。`nn.Linear`，不走 vLLM 的 `ColumnParallelLinear`。
- 实际 kernel 在 `agent_env` 的 `site-packages/nvidia/cublas/lib/libcublasLt.so.12`，名字是 `cutlass_80_tensorop_f16_s16816gemm_relu_f16_*`。仓库里没有这份 `.cu`。

## NCU 屋顶与阿姆达尔上限（只针对 TTFT GEMM）

NCU 2026.2.1，`--profile-from-start off`，只采集预填和 CLIP 的 CUTLASS GEMM、`flash_fwd_kernel`、`implicit_convolve_sgemm`。报告：`figures/ncu_ttft.ncu-rep`。图：`figures/operator-ttft-roofline.png`。屋顶比例是各管道利用率的最大值。这几块 GEMM 的最高管道是算力，DRAM 只有 9%–33%。

| 算子 | NCU 时间 | 达到屋顶 | 拉到 100% 少掉 | 对 26.22 ms TTFT kernel 的加速 |
|---|---:|---:|---:|---:|
| GEMM · 预填 MLP | 12.48 ms | 80.5% | 2.38 ms | 1.10× |
| GEMM · 预填其余 | 4.98 ms | 54.2% | 2.17 ms | 1.09× |
| GEMM · CLIP | 6.49 ms | 77.5% | 1.44 ms | 1.06× |
| FlashAttention · 预填 | 0.80 ms | 42.8% | 0.42 ms | 1.02× |
| FlashAttention · CLIP | 0.57 ms | 48.5% | 0.26 ms | 1.01× |
| Conv · CLIP | 0.08 ms | 38.6% | 0.05 ms | 1.00× |

三块 GEMM 都拉到屋顶，少 6.0 ms 量级，TTFT kernel 大约 **1.17×**（只算 MLP + CLIP 是 2.4+1.4 ms）。六个一起少 6.72 ms，26.22 ms 降到 19.50 ms，**1.34×**。客户端 TTFT 中位数 37.9 ms 上，六个一起大约 **1.22×**。128 token 的整次请求约 888 ms，同样这 6.72 ms 只有 **1.01×**。

搜索用的形状（fp16，带 bias，因为 `F.linear` 带 bias）：

- 预填 MLP：`M=259, K=2048, N=8192` 和 `M=259, K=8192, N=2048`
- 预填其余：注意力投影，`M=259`，`c_attn` 的 N 约 2304，`c_proj` 的 N、K 为 2048
- CLIP：`M=257, K=1024, N=4096` 及其转置

## 改 AutoKernel 时要动的地方

环境仍是 `autokernel/` 自己的 uv。不要跑 `profile.py`（这块卡上 PyTorch CUPTI 失败，形状已经在上面）。

- `bench.py`：`matmul` 的测试形状换成上面几组，dtype 只留 fp16，输入带 bias。对照保持 `torch.matmul` / `F.linear`（cuBLASLt）。保留条件是墙钟短于这个对照。
- `bench.py` 和 `profile.py` 的 GPU 表加上 5060 Ti：FP16 Tensor Core **47.4 TFLOPS**，带宽 **448 GB/s**。表里没有这张卡时，屋顶百分比是 0。
- `program.md`：写明基线是 `cutlass_80_tensorop_..._256x64`，预填 MLP 已经 80.5%、CLIP 77.5%、预填其余 54.2%。停的标准是超过 cuBLASLt 的墙钟，或屋顶比例不再上升。不要去搜解码 GEMV、SwiGLU、RoPE。

搜赢的 kernel 还要接回去才算数。预填 MLP 和预填其余接到 `UnquantizedLinearMethod.apply`，其余形状仍走 `F.linear`。CLIP 要另改 `clip_model.py`，否则那 6.39 ms 不会变。验收用图像 benchmark 的 TTFT。编译 vLLM 扩展时带 `TORCH_CUDA_ARCH_LIST=12.0`。没有变快就退回 `F.linear`。

## 不要让新 agent 做的事

- 不要 `pip install -e vllm` **而不加** `--no-deps`。否则会按 `torch==2.5.1` 把 cu128 换掉，sm_120 立刻坏。
- 不要为了“对齐 requirements”把 torch 退回 2.5.1，也不要重编 sm_90 来“修”5060。
- 不要把两份权重当成重复文件删掉。
- 不要再用 `benchmark_serving.py` 的 random / completions 当作 StarVector 成绩。
- 不要 `pip install -e ~/workspace/star-vector`。
- 不要把 `datasets` 升到 5.x，也不要把 `huggingface-hub` 升到 2.x。
- 不要重新打开 FlashAttention-3，除非明确要在 Hopper 上编。`setup.py` 里 FA3 扩展已去掉；`.deps/vllm-flash-attn-d4e0903/CMakeLists.txt` 里 `FA3_ENABLED` 是 OFF，并且给 sm_120 加过 arch。重编必须带：
  - `CUDA_HOME=/usr/local/cuda-13.3`
  - `TORCH_CUDA_ARCH_LIST=12.0`
  - `VLLM_CUTLASS_SRC_DIR=/home/blossonadmin/kernel-agent/vllm/.deps/cutlass-3.7.0`
  - `VLLM_FLASH_ATTN_SRC_DIR=/home/blossonadmin/kernel-agent/vllm/.deps/vllm-flash-attn-d4e0903`
  - `LIBRARY_PATH=/usr/lib/wsl/lib`
  - `pip install -e . --no-build-isolation --no-deps`
- 主 `CMakeLists.txt` 的 `CUDA_SUPPORTED_ARCHS` 已加上 `12.0`。不要改回只到 9.0，否则 5060 的 cubin 会被滤掉。
- 图像请求不要改成不带 `<image-start>` 的纯 `<svg`。chat 模板如果只输出 text part，占位符会丢，服务返回 500。
- CUDA graph 已经打开。只在 Eager 里变快的融合，不能当成 serving 加速。先在这条图像 benchmark 上复测 TTFT 和 TPOT。Eager 解码是 8.52 ms，graph 解码是 5.66 ms。
- 不要再尝试用 `torch.profiler` 的 CUDA 活动看 kernel 名。这块 sm_120 上 PyTorch 的 CUPTI 初始化失败。阶段时间看 `phase_profile.json`。graph 内部的 kernel 时间看 `figures/nsys_starvector.nsys-rep`，不要用权重字节除带宽去切 5.66 ms。
- 不要做 LayerNorm / GELU 融合来优化这条 TPOT。nsys 里它们分别是 0.14 ms 和 0.016 ms/步。
- 不要把 CLIP 塞进现在这张 decode CUDA graph。捕获输入里没有图像，解码步也不会调用 `embed_image`。
- `kernel-agent/autokernel` 是另一套环境。不要往 `agent_env` 里装它的 torch。改它的评测可以，不要用它的 `profile.py` 去扫 StarVector。
- 不要把 AutoKernel 论文里的 H100 FP16 GEMM 表当成能超过 cuBLAS。那张表是起步 kernel，相对 cuBLAS 是 0.29×–0.43×。这里的预填 MLP 已经在 cuBLASLt 的 80.5% 屋顶上。
