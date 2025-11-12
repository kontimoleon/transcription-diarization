#!/bin/bash

# Input directory (change as needed)
inputDir="./interview-videos"

# Output directory
outputDir="$inputDir/extracted-audio-wav"
# Create output directory if it doesn't exist
mkdir -p "$outputDir"

# Loop through all video files in the input directory
for inputFile in "$inputDir"/*; do
    if [[ -f "$inputFile" ]]; then  # Check if it is a file
        baseName=$(basename "$inputFile")
        fileName="${baseName%.*}"  # Get the filename without extension
        outputFile="$outputDir/$fileName.wav"
        
        echo "Extracting audio from $baseName..."
        
        # Run FFmpeg to extract audio as WAV
        ffmpeg -i "$inputFile" -vn -acodec pcm_s16le -ar 44100 -ac 2 "$outputFile" -y
    fi
done