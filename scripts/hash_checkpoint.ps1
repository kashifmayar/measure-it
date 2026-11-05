# SHA-256 for best.pt (README / assessment §5.5)
$path = Join-Path $PSScriptRoot "..\models\runs\unet_pen\best.pt"
if (-not (Test-Path $path)) { Write-Error "Missing $path"; exit 1 }
Get-FileHash -Path $path -Algorithm SHA256 | Format-List
