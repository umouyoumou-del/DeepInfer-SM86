# 把 DeepInfer-SM86 用到个人生产力

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
│ Qwen2.5-7B mixed Q4/Q3  │
│ 3-bit MLP/head          │
│ 4-bit attention/embed   │
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

| 指标 | 值 |
|---|---:|
| prefill | 约 5.8–6.3 秒 |
| decode | 约 25.7 tokens/s |
| 64-token 请求总延迟 | 约 8.4 秒 |
| 单次 24-token 快捷回答 | 约 7.3 秒 |
| context | 256 tokens（prompt + output） |

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

## 7. 与 3B 的关系

- 7B 混合：质量更好，当前网关默认，约 25.7 tokens/s decode；
- 3B 原生：约 42.7 tokens/s，但当前是测速 harness，尚未做成常驻 3B checkpoint 服务。

因此当前网关选择 7B；后续可以增加 `--checkpoint` 切换 3B/7B，或做任务路由。

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
# 1. 启动网关
python apps\server.py --max-cache 256 --port 18000

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
