import os
from pydub import AudioSegment

def combine_wavs_in_folder(folder_path, output_file):
    """Mix all .wav files in a folder and save the combined file."""
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    if not wav_files:
        print(f"No .wav files found in {folder_path}")
        return

    wav_files.sort()  # Ensure consistent order
    combined_audio = None

    # Combine all audio files
    for wav_file in wav_files:
        file_path = os.path.join(folder_path, wav_file)
        audio = AudioSegment.from_file(file_path)

        if combined_audio is None:
            combined_audio = audio
        else:
            combined_audio = combined_audio.overlay(audio)

    if combined_audio is not None and len(combined_audio) > 0:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        combined_audio.export(output_file, format="wav")
        print(f"Combined .wav saved to {output_file}")
    else:
        print(f"Skipping {folder_path}, no valid audio to save.")

def process_moises_db(root_folder, output_folder):
    """Combine .wav files from specific types of folders into categories."""
    categories = {
        "bass": ["bass"],
        "drums": ["drums", "percussion"],
        "vocals": ["vocals"],
        "other": []  # Catch-all for remaining types
    }
    # Discover all unique folder types
    unique_folders = set()
    for song_name in os.listdir(root_folder):
        song_path = os.path.join(root_folder, song_name)
        if not os.path.isdir(song_path):
            continue
        for part_name in os.listdir(song_path):
            unique_folders.add(part_name)
    # Assign remaining folders to "other"
    for folder in unique_folders:
        if not any(folder in types for types in categories.values()):
            categories["other"].append(folder)
    print("Folder categories:")
    for category, types in categories.items():
        print(f"  {category}: {types}")
    # Process each song
    for song_name in os.listdir(root_folder):
        song_path = os.path.join(root_folder, song_name)
        if not os.path.isdir(song_path):
            continue
        for category, types in categories.items():
            category_output_path = os.path.join(output_folder, song_name, f"{category}.wav")
            combined_audio = None  # Initialize as None
            for part_name in types:
                part_path = os.path.join(song_path, part_name)
                if not os.path.isdir(part_path):
                    continue
                # Combine all .wav files in this part folder
                wav_files = [f for f in os.listdir(part_path) if f.endswith('.wav')]
                wav_files.sort()
                for wav_file in wav_files:
                    file_path = os.path.join(part_path, wav_file)
                    audio = AudioSegment.from_file(file_path)
                    if combined_audio is None:
                        combined_audio = audio
                    else:
                        combined_audio = combined_audio.overlay(audio)
            if combined_audio is not None and len(combined_audio) > 0:
                os.makedirs(os.path.dirname(category_output_path), exist_ok=True)
                combined_audio.export(category_output_path, format="wav")
                print(f"Saved combined audio for {category} to {category_output_path}")
            else:
                print(f"No audio to save for {category} in song {song_name}.")







# Example usage
root_folder = r"D:\User\Desktop\musdb_normal\moisesdb_4stem_step1"  # Replace with the path to your Moises DB
output_folder = r"D:\User\Desktop\musdb_normal\moisesdb_4stem_step2"  # Replace with the path to save combined .wav files
process_moises_db(root_folder, output_folder)
