# DeepInfer-SM86 installer for Windows (an NVIDIA GPU with a CUDA 12 driver).
#
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/umouyoumou-del/DeepInfer-SM86/main/install.ps1 | iex"
#
# Puts the commands deepinfer, deepinfer-cli, deepinfer-server and deepinfer-bench in
# %LOCALAPPDATA%\deepinfer\bin (added to your PATH), backed by a private Python environment
# next to it; nothing else on the system changes.  Run it again to update.  Optional settings,
# as environment variables (see install.sh): DEEPINFER_HOME, DEEPINFER_BIN, DEEPINFER_REF,
# DEEPINFER_SOURCE, PYTHON, TORCH_INDEX_URL, DEEPINFER_NO_MODIFY_PATH.

& {
    $ErrorActionPreference = "Stop"
    $ProgressPreference = "SilentlyContinue"  # Invoke-WebRequest is much slower drawing progress
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    function Say([string]$Text) { Write-Host "deepinfer: $Text" }

    function Setting([string]$Name, [string]$Default) {
        $value = [Environment]::GetEnvironmentVariable($Name)
        if ($value) { return $value }
        return $Default
    }

    # run a program, stop when it fails
    function Run([string]$Exe, [string[]]$Arguments) {
        & $Exe @Arguments
        if ($LASTEXITCODE -ne 0) { throw "deepinfer: '$Exe $($Arguments -join ' ')' failed with exit code $LASTEXITCODE" }
    }

    # a program's output, or $null when it is missing or fails
    function Output([string]$Exe, [string[]]$Arguments) {
        $saved = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $out = & $Exe @Arguments 2>$null
            if ($LASTEXITCODE -eq 0 -and $out) { return ($out -join "`n").Trim() }
        } catch {
        } finally {
            $ErrorActionPreference = $saved
        }
        return $null
    }

    # whether a Python imports a CUDA build of PyTorch 2.5 or newer; "base": and is not a venv itself
    function Test-CudaTorch([string]$Python, [string]$Mode) {
        $probe = "import sys, torch; v = tuple(int(p) for p in torch.__version__.split('+')[0].split('.')[:2]); " +
            "venv = sys.argv[-1] == 'base' and sys.prefix != sys.base_prefix; " +
            "print('yes' if torch.version.cuda and v >= (2, 5) and not venv else '')"
        return [bool](Output $Python @("-c", $probe, $Mode))
    }

    $repo = "umouyoumou-del/DeepInfer-SM86"
    $ref = Setting "DEEPINFER_REF" "main"
    $root = Setting "DEEPINFER_HOME" (Join-Path $env:LOCALAPPDATA "deepinfer")
    $bin = Setting "DEEPINFER_BIN" (Join-Path $root "bin")
    $venv = Join-Path $root "venv"
    $commands = "deepinfer", "deepinfer-cli", "deepinfer-server", "deepinfer-bench"

    New-Item -ItemType Directory -Force -Path $root | Out-Null
    $vpy = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path $vpy)) {  # an update reuses the environment and needs no other Python
        $probe = "import sys; print(sys.executable if sys.version_info >= (3, 10) else '')"
        $candidates = @()
        if ($env:PYTHON) {
            $candidates += , @($env:PYTHON)
        } else {
            $candidates += , @("py", "-3.12")
            $candidates += , @("py", "-3.11")
            $candidates += , @("py", "-3.10")
            $candidates += , @("python")
            $candidates += , @("python3")
        }
        $python = $null
        foreach ($candidate in $candidates) {
            $rest = @($candidate | Select-Object -Skip 1)
            $python = Output $candidate[0] ($rest + @("-c", $probe))
            if ($python) { break }
        }
        if (-not $python) { throw "deepinfer: no Python 3.10 or newer found: install one from https://www.python.org/downloads/ or set PYTHON" }
        $arguments = @("-m", "venv", $venv)
        if (Test-CudaTorch $python "base") {
            Say "creating $venv, sharing the PyTorch of $python"
            $arguments = @("-m", "venv", "--system-site-packages", $venv)
        } else {
            Say "creating $venv with $python"
        }
        try { Run $python $arguments } catch { Remove-Item -Recurse -Force $venv -ErrorAction SilentlyContinue; throw }
    }
    Run $vpy @("-m", "pip", "install", "--quiet", "--upgrade", "pip")

    if (-not (Test-CudaTorch $vpy "")) {
        $index = Setting "TORCH_INDEX_URL" ""
        if (-not $index) {
            $smi = Output "nvidia-smi" @()
            if ($smi -match "CUDA Version:\s*(\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -lt 12) {
                    throw "deepinfer: the NVIDIA driver supports CUDA $major.$minor; DeepInfer needs CUDA 12 (driver 528 or newer): update the driver or set TORCH_INDEX_URL"
                } elseif ($major -gt 12 -or $minor -ge 8) { $tag = "cu128" }
                elseif ($minor -ge 6) { $tag = "cu126" }
                elseif ($minor -ge 4) { $tag = "cu124" }
                else { $tag = "cu121" }
                $index = "https://download.pytorch.org/whl/$tag"
            } else {
                Say "warning: no NVIDIA driver found (nvidia-smi); the commands install, but running a model needs an NVIDIA GPU"
            }
        }
        if ($index) {
            Say "installing PyTorch from $index (about 3 GB)"
            Run $vpy @("-m", "pip", "install", "torch>=2.5", "--index-url", $index)
        }
    }

    $source = Setting "DEEPINFER_SOURCE" ""
    $zip = $null
    if (-not $source) {
        Say "downloading $repo@$ref"
        $zip = Join-Path ([IO.Path]::GetTempPath()) ("DeepInfer-SM86-" + [guid]::NewGuid().ToString("N") + ".zip")
        Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/$repo/archive/$ref.zip" -OutFile $zip
        $source = $zip
    }
    if ($source -match "://") { $url = $source }
    else { $url = ([Uri](Resolve-Path -LiteralPath $source).ProviderPath).AbsoluteUri }
    try {
        Say "installing deepinfer-sm86 and its dependencies"
        Run $vpy @("-m", "pip", "install", "--no-cache-dir", "deepinfer-sm86[cuda] @ $url")  # a rebuilt archive keeps its name
    } finally {
        if ($zip) { Remove-Item -Force $zip -ErrorAction SilentlyContinue }
    }

    New-Item -ItemType Directory -Force -Path $bin | Out-Null
    foreach ($name in $commands) {
        # pip's launchers name their Python by absolute path, so a copy runs from anywhere
        Copy-Item -Force (Join-Path $venv "Scripts\$name.exe") (Join-Path $bin "$name.exe")
    }
    if (-not (Setting "DEEPINFER_NO_MODIFY_PATH" "")) {
        $key = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey("Environment", $true)
        $old = [string]$key.GetValue("Path", "", [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        if (($old -split ";") -notcontains $bin) {
            $new = if ($old) { $old.TrimEnd(";") + ";" + $bin } else { $bin }
            $key.SetValue("Path", $new, [Microsoft.Win32.RegistryValueKind]::ExpandString)
            # a change made through .NET tells Explorer, so new terminals get the new PATH
            [Environment]::SetEnvironmentVariable("DEEPINFER_INSTALLING", "1", "User")
            [Environment]::SetEnvironmentVariable("DEEPINFER_INSTALLING", $null, "User")
            Say "added $bin to your PATH"
        }
        $key.Close()
    }
    if (($env:Path -split ";") -notcontains $bin) { $env:Path = "$bin;$env:Path" }

    & (Join-Path $bin "deepinfer.exe") info
    Write-Host ""
    Write-Host "deepinfer-sm86 is installed: $($commands -join ', ') ($bin)"
    Write-Host "  deepinfer-cli -hf coder                      chat in the terminal (the first run downloads 10 GB)"
    Write-Host "  deepinfer-cli -m model.gguf -p `"a question`"  one answer from a local GGUF"
    Write-Host "  deepinfer-server -hf coder                   page and API at http://127.0.0.1:8080/"
    Write-Host "  deepinfer-bench -hf coder -p 512,4096        prompt and generation speed"
}
