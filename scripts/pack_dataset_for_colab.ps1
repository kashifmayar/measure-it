# Zip for Colab: extracts to measure-it/dataset/images and measure-it/dataset/masks
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "dataset_colab_upload.zip"
$images = Join-Path $root "dataset\images"
$masks = Join-Path $root "dataset\masks"
if (-not (Test-Path $images)) { Write-Error "Missing $images"; exit 1 }
$staging = Join-Path $env:TEMP "measure-it-colab-pack"
Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
$destImg = Join-Path $staging "dataset\images"
$destMsk = Join-Path $staging "dataset\masks"
New-Item -ItemType Directory -Path $destImg, $destMsk -Force | Out-Null
Copy-Item "$images\*.jpg" $destImg
Copy-Item "$masks\*.png" $destMsk
Remove-Item $out -Force -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $staging "dataset") -DestinationPath $out -Force
Remove-Item $staging -Recurse -Force
Write-Host "Created $out"
Write-Host "In Colab: extract zip into /content/measure-it (repo root), then validate"
