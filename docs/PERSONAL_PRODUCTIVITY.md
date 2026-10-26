# 把 DeepInfer-SM86 用到个人生产力

> 这里的 native 指 DeepInfer SM86 native 路径，不是 Ollama 或兼容层。

这不是“再跑一次 benchmark”，而是把 4 GB 上的本地模型变成一个每天可以调用的工具。

## 1. 最终形态

```text
┌─────────────────────────┐
│ Chatbox / Obsidian /    │
│ VS Code / 浏览器插件     │
│ 或 desktop\quick_ask.ps1│
└────────────┬────────────┘
             │ OpenAI HTTP / SSE
┌────────────▼────────────┐
│ apps/server.py          │
│ /v1/models              │
│ /v1/chat/completions    │
└────────────┬────────────┘
             │ resident model
┌────────────▼────────────┐
│ 3B W4A16（默认快）       │
│ 或 7B mixed Q4/Q3       │
│ static KV + CUDA Graph  │
│ RTX 3050 4 GB           │
└─────────────────────────┘
```

模型常驻在网关进程里。桌面脚本、CLI、OpenAI 客户端都只发 HTTP 请求，不重复加载模型。

## 2. 启动

```powershell
cd D:\DeepInfer-SM86

F:\QwenQLoRA\venv\Scripts\python.exe `
  apps\server.py `
  --checkpoint D:\Qwen7B-Mixed-SM86 `
  --max-cache 256 `
  --port 18000
```

不要用 8000 端口作为固定值：本机 8000 已被 MaiBot 占用。网关默认使用
`18000`。

健康检查：

```powershell
curl.exe http://127.0.0.1:18000/health
```

## 3. OpenAI 客户端

Base URL：

```text
http://127.0.0.1:18000/v1
```

模型名：

```text
deepinfer-sm86-7b-mixed
```

支持：

- `GET /v1/models`
- `POST /v1/chat/completions`
- `stream: true` 的 SSE 流式输出
- `messages` 里的 `system` / `user` / `assistant`
- EOS 停止
- 单请求串行

## 4. 桌面快捷操作

先把要处理的文字复制到剪贴板，然后：

```powershell
.\desktop\quick_ask.ps1 -Task 总结
.\desktop\quick_ask.ps1 -Task 改写
.\desktop\quick_ask.ps1 -Task 翻译成英文
.\desktop\quick_ask.ps1 -Task 解释
.\desktop\quick_ask.ps1 -Task 提取待办
```

结果会写回剪贴板。

推荐用 PowerToys Keyboard Manager 绑定：

| 快捷键 | 动作 |
|---|---|
| `Win+Alt+Q` | 总结 |
| `Win+Alt+R` | 改写 |
| `Win+Alt+T` | 翻译成英文 |
| `Win+Alt+E` | 解释 |
| `Win+Alt+D` | 提取待办 |

PowerShell 脚本保存为 UTF-8 with BOM，否则 Windows PowerShell 5.1 会把中文
参数读成乱码。

## 5. 实测体验

在 4 GB 3050、MaiBot 同时占用后台显存的环境下：

| 后端 | prefill | decode | 一次请求 |
|---|---:|---:|---:|
| 3B DeepInfer SM86 native W4A16 | 约 0.46 秒 | 约 44.8 tokens/s | 约 0.90 秒 |
| 7B DeepInfer SM86 native（隔离） | 约 3.06 秒 | 约 25.96 tokens/s | 约 6–7 秒 |
| 7B DeepInfer SM86 native（paging） | 53 秒以上 | 约 25.1 tokens/s | 不建议日常 |
| Ollama 3B Q4_K_M | 约 0.06 秒 | 约 57.3 tokens/s | 约 0.46 秒 |
| context | 256（native）/ 4096（Ollama） | | |

时间主要花在 prefill，而不是 decode。对选中段落的总结、改写、翻译，这个延迟
可以接受；对长文总结，需要分段。

## 6. 长文本策略

当前网关上限是 256 token，不要直接塞整篇文章。

建议：

1. 先按段落切成 2–5 段；
2. 每段单独总结；
3. 把分段摘要再合并成最终摘要。

下一步可以把这个 map-reduce 流程放进 `apps/long_text.py`，并接入本地
Markdown/PDF 索引。

## 7. 3B / 7B / Ollama 怎么选

- **3B DeepInfer SM86 native**：当前个人快捷操作的默认推荐，约 0.9 秒一次请求；
- **7B DeepInfer SM86 native**：质量更高，但 4 GB 上很容易 paging，建议只在高价值短任务使用；
- **Ollama 3B**：同协议下最快，约 0.46 秒；但这是 Ollama 引擎，不是 DeepInfer 原生；
- **DeepInfer 3B DeepInfer SM86 native** 约 44.8 tokens/s，已经进入可用区间；与 Ollama 的差距主要在
  prefill 和完整 engine，而不是模型权重本身。

完整数据见：

- `docs/ARTICLE_2_DEEPINFER_SM86_PRODUCTIVITY.md`
- `docs/PRODUCTIVITY_BENCHMARK_3B_7B_OLLAMA.md`
- `docs/results/productivity_backends_combined.json`

## 8. 隐私与安全

- 请求只发到 `127.0.0.1`；
- 不经过云端 API；
- 但网关本身没有鉴权，默认只监听本机；
- 不要绑定 `0.0.0.0` 到公网；
- 剪贴板快捷操作会把选中文本写进本地 HTTP 请求；
- 邮件/日历等写操作尚未接入；接入时必须先预览再确认。

## 9. 已知限制

- `full_engine=false`，不是完整 DeepInfer engine；
- 单请求串行，不支持并发；
- 256 context，未接入长文 map-reduce；
- 3-bit MLP/head 有损，输出不保证与 BF16 等价；
- 512-context 的 512-thread GQA 路径在当前 runtime 中曾出现输出退化，已经收口到 256；
- Python 3.10 GPU harness 验证，Python 3.12 package 验证未完成；
- 没有工具调用、RAG、邮件/日历自动化。

## 10. 最小验收

```powershell
# 1. 启动 3B 网关
python apps\server.py --checkpoint D:\Qwen3B-W4A16-SM86 --backend w4a16 --max-cache 256 --port 18001

# 1b. 或启动 7B 网关
python apps\server.py --checkpoint D:\Qwen7B-Mixed-SM86 --backend mixed --max-cache 256 --port 18000

# 2. 健康检查
curl.exe http://127.0.0.1:18000/health

# 3. 非流式
python apps\ask.py --server http://127.0.0.1:18000 "用一句话解释 CUDA Graph。"

# 4. 流式
curl.exe -N http://127.0.0.1:18000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"deepinfer-sm86-7b-mixed\",\"messages\":[{\"role\":\"user\",\"content\":\"写三句话解释 CUDA Graph\"}],\"max_tokens\":64,\"stream\":true}"

# 5. 剪贴板
.\desktop\quick_ask.ps1 -Task 总结
```

全部通过，才算“从跑分成果变成个人生产力工具”。
