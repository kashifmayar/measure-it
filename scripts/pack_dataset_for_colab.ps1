# Zip dataset/images and dataset/masks for Colab upload
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path "$root\dataset\images")) { $root = "E:\Project\measure it" }
$out = Join-Path $root "dataset_colab_upload.zip"
$images = Join-Path $root "dataset\images"
$masks = Join-Path $root "dataset\masks"
if (-not (Test-Path $images)) { Write-Error "Missing $images"; exit 1 }
$staging = Join-Path $env:TEMP "measure-it-colab-pack"
Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path "$staging\images", "$staging\masks" | Out-Null
Copy-Item "$images\*.jpg" "$staging\images\"
Copy-Item "$masks\*.png" "$staging\masks\"
Compress-Archive -Path "$staging\*" -DestinationPath $out -Force
Remove-Item $staging -Recurse -Force
Write-Host "Created $out"
Write-Host "Upload to Colab or Google Drive (see notebooks/README.md)"
