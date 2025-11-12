# Input directory (change as needed)
$inputDir = ".\interview-videos"

# Output directory (create it if it doesn't exist)
$outputDir = Join-Path $inputDir "extracted-audio-wav"
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir | Out-Null
}

# Get all video files in the directory
Get-ChildItem -Path $inputDir -File | ForEach-Object {
    $inputFile = $_.FullName
    $baseName = $_.BaseName
    $outputFile = Join-Path $outputDir "$baseName.wav"

    Write-Host "Extracting audio from $($_.Name)..."

    # Run FFmpeg to extract audio as WAV
    ffmpeg -i $inputFile -vn -acodec pcm_s16le -ar 44100 -ac 2 $outputFile -y
}
