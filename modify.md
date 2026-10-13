# 这次改了什么

只加了测量，没有改计算、没有换 kernel。计时默认关闭，只有请求打到 `/start_profile` 之后才记。`/stop_profile` 会拆掉 hook 并把结果写成 JSON。

`torch.profiler` 在 RTX 5060 Ti（sm_120）上 CUPTI 报 `CUPTI_ERROR_INVALID_DEVICE`，trace 里没有 CUDA kernel。阶段和叶子时间改用 CUDA Event。

## 新文件

`vllm/vllm/model_executor/phase_profile.py`

- `phase(name)`：在当前 CUDA stream 上包一段 Event。未启用时直接 yield。
- `attach(model)` / `detach()`：给叶子模块挂 forward hook。覆盖 vLLM 的 `ColumnParallelLinear`、`RowParallelLinear`、`Attention`、`ParallelLMHead`，以及 `nn.Linear`、`nn.Conv2d`、`nn.LayerNorm`、`nn.MultiheadAttention`、名字里带 GELU 的激活。
- hook 记下模块全名、类型和当时的阶段栈。CUDA graph 回放不进 Python，所以 graph 解码没有叶子记录。
- `dump(path)`：先 `synchronize`，再按阶段、区域（vision / adapter / llm）、角色（mlp_linear、attention、norm 等）汇总，写入 JSON。

## 改过的文件

`vllm/vllm/worker/worker.py`

- profiler 增加 `record_shapes=True`。`VLLM_PROFILER_WITH_STACK=0` 时关掉 `with_stack`，避免 trace 过大。
- `start_profile` 在原来的 `profiler.start()` 之前 `attach` 模型并打开 Event 计时。
- `stop_profile` 在 `profiler.stop()` 之后把 `phase_profile.json` 写到 `VLLM_TORCH_PROFILER_DIR`，然后 `detach`。

`vllm/vllm/worker/model_runner.py`

- `execute_model` 里，模型前向包了 `starvector.prefill` 或 `starvector.decode`（`record_function` 加 `phase`）。
- `compute_logits` 和 `sample` 包了 `starvector.sample`。
- 这些标记在 profiler 没开时，`phase()` 是空操作。`record_function` 仍在，开销很小。

`vllm/vllm/model_executor/models/starvector.py`

- `embed_image` 包了 `starvector.vision`。范围是 CLIP 前向加 Adapter，不含外面的 token 拼接。

`vllm/benchmarks/test_benchmark_serving.py`

- 原来的 `--profile` 用 chat 请求去打 `/start_profile`，openai-chat 后端会因为 URL 不是 `chat/completions` 而失败。改成对这两个端点直接 POST。
- 新增 `--profile-trace-dir`。停表之后读目录里最新的 Chrome trace，并打印 `phase_profile.json`。
- trace 扫描会跳过 `phase_profile.json`。没有 CUDA kernel 时会说明是 CUPTI 失败，而不是把空表当成「没有 GPU 工作」。

## 怎么得到数字

服务需要在启动前设置 `VLLM_TORCH_PROFILER_DIR`。客户端：

```bash
python test_benchmark_serving.py \
  --backend openai-chat --host 127.0.0.1 --port 8000 \
  --endpoint /v1/chat/completions \
  --model /home/blossonadmin/kernel-agent/vllm/models/starvector-1b-im2svg \
  --tokenizer /home/blossonadmin/kernel-agent/vllm/models/starvector-1b-im2svg \
  --trust-remote-code --dataset-name local-images \
  --num-prompts 1 --max-concurrency 1 --random-output-len 32 \
  --ignore-eos --profile \
  --profile-trace-dir "$VLLM_TORCH_PROFILER_DIR"
```

Eager 对照是同一条命令，服务加上 `--enforce-eager`，输出长度用过 16。测完的服务已经停掉。阶段时间在 `profile.md` 的「阶段拆分」。graph 内部的 kernel 时间在「nsys kernel 时间」，那一节覆盖了按字节估算的算子拆分。

## 为 nsys 加的测量

仍然不改计算。`STARVECTOR_NVTX` 和 `STARVECTOR_NSYS` 默认都没设置，平时服务不打范围、也不调用 profiler API。

`vllm/vllm/model_executor/phase_profile.py`

- `phase(name)` 在 `STARVECTOR_NVTX=1` 时额外 `nvtx.range_push/pop`，名字是 `prefill`、`decode`、`vision`、`sample`。Event 计时仍只在 `/start_profile` 把 `_enabled` 打开之后才记。

`vllm/vllm/worker/worker.py`

- `STARVECTOR_NSYS=1` 时，`/start_profile` 调 `cudaProfilerStart`，`/stop_profile` 先 `synchronize` 再 `cudaProfilerStop`。可以不设 `VLLM_TORCH_PROFILER_DIR`，这样不会同时开 PyTorch profiler。

`vllm/vllm/entrypoints/openai/api_server.py`

- `/start_profile` 和 `/stop_profile` 在 `STARVECTOR_NSYS=1` 时也会注册。原先只看 `VLLM_TORCH_PROFILER_DIR`。

`vllm/vllm/worker/model_runner.py`

- 包 `sample` 时弄丢了 `observability_config` 那段 `if` 的条件头，导入即 `SyntaxError`。已经补回。服务在没开 observability 时不会走进这段。

采集命令（服务进程由 nsys 拉起，热身在 start 之前，只记后面那一条图像请求）：

```bash
export STARVECTOR_NVTX=1
export STARVECTOR_NSYS=1
export PYTHONPATH=/home/blossonadmin/workspace/star-vector:$PYTHONPATH
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --cuda-trace-scope=process-tree \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --sample=none --force-overwrite=true \
  -o /home/blossonadmin/kernel-agent/figures/nsys_starvector \
  vllm serve /home/blossonadmin/kernel-agent/vllm/models/starvector-1b-im2svg \
  --trust-remote-code --dtype float16 --max-model-len 4096 \
  --gpu-memory-utilization 0.85 --port 8000
```

客户端仍用 `test_benchmark_serving.py --profile`，1 张图、`--random-output-len 32`。`--profile` 会先发一条不采集的热身，再 POST `/start_profile`。汇总：

```bash
nsys stats --report nvtx_kern_sum --report cuda_gpu_kern_sum \
  figures/nsys_starvector.nsys-rep
```

`vision` 嵌在 `prefill` 里面。`nvtx_kern_sum` 的 prefill 行包含 vision 的 kernel，两边相加会重复计算。解码 GEMV 用 grid 区分：`(2048,16)` 与 `(512,32)` 是 MLP，`(576,16)` 与 `(512,16)` 是注意力投影，`(12304,16)` 是词表。柱状图是 `figures/operator-time-share.png`。采集用的服务已经停掉。
