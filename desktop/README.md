# Desktop quick actions

Start the gateway first:

```powershell
cd D:\DeepInfer-SM86
python apps\server.py --max-cache 256 --port 18000
```

Then run a clipboard action:

```powershell
.\desktop\quick_ask.ps1 -Task 总结
.\desktop\quick_ask.ps1 -Task 翻译成英文
.\desktop\quick_ask.ps1 -Task 提取待办
```

The result is written back to the clipboard.

## Hotkey

Use PowerToys Keyboard Manager (or AutoHotkey) to bind a shortcut to:

```powershell
powershell -ExecutionPolicy Bypass -File D:\DeepInfer-SM86\desktop\quick_ask.ps1 -Task 总结
```

Suggested shortcuts:

- `Win+Alt+Q`: 总结
- `Win+Alt+R`: 改写
- `Win+Alt+T`: 翻译成英文
- `Win+Alt+E`: 解释

This is intentionally a thin client: the model stays resident in the gateway,
so each quick action pays only prefill + decode, not model load time.
