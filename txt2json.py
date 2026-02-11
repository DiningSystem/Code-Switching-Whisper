import os
import json


def read_txt_folder_to_json(folder_path):
    result = {}

    # Loop through all files in folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines()]

                if len(lines) >= 3:
                    code_switch = lines[0]
                    vi_full = lines[1]
                    en_full = lines[2]

                    # Convert .txt → .mp3
                    wav_name = filename.replace(".txt", ".wav")

                    result[wav_name] = {
                        "code_switch": code_switch,
                    }

    return result


if __name__ == "__main__":
    folder_path = "./Data"  # change this
    output = read_txt_folder_to_json(folder_path)

    # Print JSON nicely formatted
    with open("./transcript.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
