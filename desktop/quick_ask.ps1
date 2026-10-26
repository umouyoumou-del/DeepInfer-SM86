param(
    [ValidateSet("总结", "改写", "翻译成英文", "解释", "提取待办")]
    [string]$Task = "总结",
    [string]$Server = "http://127.0.0.1:18000",
    [int]$MaxTokens = 512
)

$ErrorActionPreference = "Stop"
$text = Get-Clipboard -Raw
if ([string]::IsNullOrWhiteSpace($text)) {
    throw "Clipboard is empty. Select/copy text first."
}

$prompt = @"
请对下面的内容执行「$Task」，只输出最终结果，不要重复任务说明：

$text
"@

$body = @{
    model = "deepinfer-sm86-7b-mixed"
    messages = @(@{ role = "user"; content = $prompt })
    max_tokens = $MaxTokens
    temperature = 0.0
    stream = $false
} | ConvertTo-Json -Depth 8

$response = Invoke-RestMethod `
    -Uri "$($Server.TrimEnd('/'))/v1/chat/completions" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body)) `
    -TimeoutSec 600

$answer = [string]$response.choices[0].message.content
Set-Clipboard -Value $answer
Write-Host "Result copied to clipboard:" -ForegroundColor Green
Write-Host $answer
