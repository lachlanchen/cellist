import os

# Specify the directories to be created
directories = [
    "annotation_algorithm",
    "cropped",
    "images",
    "models",
    "models_backup",
    "temp",
    "uploads",
    "annotation_manual",
    "dataset",
]

# Get the current working directory
cwd = os.getcwd()

# Iterate over the directories to be created
for directory in directories:
    # Construct the full path
    dir_path = os.path.join(cwd, directory)

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# List the contents of the current directory to verify that the directories were created
os.listdir(cwd)

