import os
import shutil

# Set the paths
source_folder = r"D:\User\Desktop\musdb_normal\dataasets\musdb18\train"  # Path to the musdb train folder
destination_folder = r"D:\User\Desktop\musdb_normal\files"  # Folder where you want to copy the bass files
# Ensure the destination folder exists
os.makedirs(destination_folder, exist_ok=True)
# Counter for renaming files
file_counter = 4001
# Walk through each folder in the source directory
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file == 'vocals.wav':  # Check if the file is 'bass.mp3'
            source_file_path = os.path.join(root, file)  # Get the full path to the source file
            new_file_name = f"{file_counter}.wav"  # Create a new name for the file (e.g., 1001.mp3)
            destination_file_path = os.path.join(destination_folder,new_file_name)  # Full path for the destination file
            # Copy the file to the new folder with the new name
            shutil.copy(source_file_path, destination_file_path)
            print(f"Copied: {source_file_path} -> {destination_file_path}")
            # Increment the counter for the next file
            file_counter += 1
print("All bass.wav files have been copied and renamed.")
