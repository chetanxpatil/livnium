"""
experiment/prep_data.py — Data preparation utility for the pure collapse experiment.

Symlinks or copies the compiled conversation context data from the main
chat directory to the local experiment folder.
"""

import os
import shutil

def main():
    src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../chat/data"))
    dst_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    
    src_file = os.path.join(src_dir, "chat_context.tsv")
    dst_file = os.path.join(dst_dir, "chat_context.tsv")
    
    print(f"checking source data: {src_file}")
    if not os.path.exists(src_file):
        print(f"Error: context data not found at {src_file}.")
        print("Please run prep_chat_context.py inside the chat/ folder first.")
        return
        
    os.makedirs(dst_dir, exist_ok=True)
    
    # Create symlink or fallback to copy
    if os.path.exists(dst_file):
        os.remove(dst_file)
        
    try:
        os.symlink(src_file, dst_file)
        print(f"successfully linked data: {dst_file} -> {src_file}")
    except OSError:
        shutil.copy2(src_file, dst_file)
        print(f"successfully copied data to: {dst_file}")

if __name__ == "__main__":
    main()
