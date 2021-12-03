import glob
import os
import shutil


def get_all_cache_folders(src="."):
    return glob.glob(os.path.join(src, "**", "__pycache__"), recursive=True)


if __name__ == "__main__":
    folders = get_all_cache_folders()

    if len(folders) == 0:
        print("No folder matching __pycache__, exiting...")

    print("Here are the folders that will be deleted:")
    print("\n".join(folders))
    proceed = input("Proceed ? [Y/N]: ").upper()
    if proceed == "Y":
        print("Deleting folders")
        for folder in folders:
            shutil.rmtree(folder)
    else:
        print("Exiting without deleting folders")
