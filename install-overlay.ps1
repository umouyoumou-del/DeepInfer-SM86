param(
    [Parameter(Mandatory = $true)]
    [string]$DeepInferRoot
)

$ErrorActionPreference = "Stop"

$root = (Resolve-Path $PSScriptRoot).Path
$target = (Resolve-Path $DeepInferRoot).Path
$targetPackage = Join-Path $target "deepinfer"

if (-not (Test-Path $targetPackage)) {
    throw "No deepinfer package found at $targetPackage"
}

Write-Host "Overlaying DeepInfer-SM86 into $target"
Copy-Item -Path (Join-Path $root "deepinfer\*") -Destination $targetPackage -Recurse -Force
Copy-Item -Path (Join-Path $root "tests\*") -Destination (Join-Path $target "tests") -Recurse -Force
Copy-Item -Path (Join-Path $root "scripts\*") -Destination (Join-Path $target "scripts") -Recurse -Force
Write-Host "Done. Back up the target checkout before using this in a real project."
