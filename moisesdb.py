import os
from pydub import AudioSegment


def combine_wavs_in_folder(folder_path, output_path):
    """Mix all .wav files in a folder and save the combined file."""
    # List all .wav files in the folder
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    if not wav_files:
        print(f"No .wav files found in {folder_path}")
        return

    # Sort files to ensure consistent order (optional)
    wav_files.sort()

    # Load the first file as the base
    combined = AudioSegment.from_file(os.path.join(folder_path, wav_files[0]))

    # Overlay all other .wav files onto the base
    for wav_file in wav_files[1:]:
        file_path = os.path.join(folder_path, wav_file)
        audio = AudioSegment.from_file(file_path)
        combined = combined.overlay(audio)

    # Export the combined file
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"combined.wav")
    combined.export(output_file, format="wav")
    print(f"Combined .wav saved to {output_file}")


def process_moises_db(root_folder, output_folder):
    """Iterate through the Moises DB and mix .wav files in each part folder."""
    for song_name in os.listdir(root_folder):
        song_path = os.path.join(root_folder, song_name)
        if not os.path.isdir(song_path):
            continue

        for part_name in os.listdir(song_path):
            part_path = os.path.join(song_path, part_name)
            if not os.path.isdir(part_path):
                continue

            output_path = os.path.join(output_folder, song_name, part_name)
            combine_wavs_in_folder(part_path, output_path)


# Example usage
root_folder = r"D:\User\Desktop\musdb_normal\moisesdb_v0.1" # Replace with the path to your Moises DB
output_folder = r"D:\User\Desktop\musdb_normal\moisesdb_4stem_step1"  # Replace with the path to save combined .wav files
process_moises_db(root_folder, output_folder)
