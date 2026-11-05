# Smaller Colab zip: JPEG quality 85, max long edge 1024 (faster upload, still trains)
param([int]$MaxSide = 1024, [int]$Quality = 85)

$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "dataset_colab_upload_small.zip"
$images = Join-Path $root "dataset\images"
$masks = Join-Path $root "dataset\masks"
$staging = Join-Path $env:TEMP "measure-it-colab-small"
Remove-Item $staging -Recurse -Force -ErrorAction SilentlyContinue
$destImg = Join-Path $staging "dataset\images"
$destMsk = Join-Path $staging "dataset\masks"
New-Item -ItemType Directory -Path $destImg, $destMsk -Force | Out-Null

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$resizeScript = @"
import sys
from pathlib import Path
import cv2
root = Path(r'$root')
staging = Path(r'$staging')
max_side = $MaxSide
quality = $Quality
for src in sorted((root / 'dataset' / 'images').glob('*.jpg')):
    bgr = cv2.imread(str(src))
    if bgr is None:
        continue
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(Path(r'$destImg') / src.name), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
for src in sorted((root / 'dataset' / 'masks').glob('*.png')):
    m = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
    if m is None:
        continue
    h, w = m.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        m = cv2.resize(m, (int(w * s), int(h * s)), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(Path(r'$destMsk') / src.name), m)
print('done', len(list(Path(r'$destImg').glob('*.jpg'))))
"@
& $py -c $resizeScript
Compress-Archive -Path (Join-Path $staging "dataset") -DestinationPath $out -Force
Remove-Item $staging -Recurse -Force
Write-Host "Created $out (smaller — upload this to Drive, not full-res zip)"
