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
foreach ($name in @("tests", "scripts", "apps", "desktop")) {
    $source = Join-Path $root $name
    if (Test-Path $source) {
        $destination = Join-Path $target $name
        New-Item -ItemType Directory -Force -Path $destination | Out-Null
        Copy-Item -Path (Join-Path $source "*") -Destination $destination -Recurse -Force
    }
}
Write-Host "Done. Back up the target checkout before using this in a real project."
