input_dir="./Data2"
output_dir="./wav_data"

find "$input_dir" -type f -name "*.wav" | while read -r file; do
    rel_path="${file#$input_dir/}"              # remove input prefix
    rel_dir="$(dirname "$rel_path")"            # subdirectory
    base="$(basename "$file")"
    newname="${base%%_*}.wav"

    mkdir -p "$output_dir/$rel_dir"

    ffmpeg -y -i "$file" -c:a pcm_s16le \
        "$output_dir/$rel_dir/$newname"
done
