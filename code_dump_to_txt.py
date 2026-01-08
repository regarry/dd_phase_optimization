import os

# --- CONFIGURATION ---

# 1. Folders to skip completely
IGNORED_DIRS = {
    # Standard ignores
    '.git', 'node_modules', 'venv', '__pycache__', '.idea', '.vscode', 
    'dist', 'build', 'coverage',
    
    # YOUR CUSTOM IGNORES
    'beads_img_defocus',
    'output_layer_imgs',
    'results',
    'traininglocations',
    'training_results',
    'inference_results',
    'beam_profiles',
    'logs',
    'psf_images'
}

# 2. File extensions to skip
IGNORED_EXTENSIONS = {
    # Standard ignores
    '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', 
    '.mp4', '.mp3', '.pdf', '.zip', '.tar', '.gz', 
    '.class', '.dll', '.exe', '.o', '.so',
    '.DS_Store', '.lock','.yml','.yaml',
    
    # Text/Doc files you want to ignore
    '.txt', '.md',
    
    # YOUR CUSTOM IGNORES
    '.pyc', '.log'
}

# 3. SPECIFIC EXCLUDES (Files to skip by name, regardless of extension)
# Example: 'secret_key.py', 'old_test.py'
SPECIFIC_EXCLUDES = {
    'cuda_error.py',
    'fwhm.py',
    'code_dump_to_txt.py',
    'notes',
    'physics_utils_copy.py',
    'cuda_debug.sh',
    'cnn_utils.py',
    'FWHM_cal.py',
    '.gitignore'
    # Add filenames here...
}

# 4. SPECIFIC INCLUDES (Files to include even if their extension is ignored)
# Example: 'README.md' (even though .md is ignored above), 'important_log.log'
SPECIFIC_INCLUDES = {
    'config.yaml'
    # Add filenames here...
}

OUTPUT_FILE = 'codebase_dump.txt'

def is_text_file(filepath):
    """Simple check to see if a file is text (not binary)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            f.read(1024)
        return True
    except (UnicodeDecodeError, IOError):
        return False

def main():
    # Use current directory by default if input is empty
    root_dir = input(f"Enter path (Press Enter for current dir): ").strip()
    if not root_dir:
        root_dir = os.getcwd()

    if not os.path.exists(root_dir):
        print("Error: Directory does not exist.")
        return

    print(f"Scanning '{root_dir}'...")
    
    file_count = 0
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for subdir, dirs, files in os.walk(root_dir):
            # Modify 'dirs' in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            
            for file in files:
                file_path = os.path.join(subdir, file)
                rel_path = os.path.relpath(file_path, root_dir)
                _, ext = os.path.splitext(file)

                # --- LOGIC START ---
                
                # 1. Check Specific Excludes (Highest Priority Skip)
                if file in SPECIFIC_EXCLUDES:
                    continue

                should_process = False

                # 2. Check Specific Includes (Override Extension Rules)
                if file in SPECIFIC_INCLUDES:
                    should_process = True
                
                # 3. Standard Checks (If not forced include)
                else:
                    is_ignored_ext = ext.lower() in IGNORED_EXTENSIONS
                    is_output_file = (file == OUTPUT_FILE)
                    
                    if not is_ignored_ext and not is_output_file:
                        should_process = True

                # --- LOGIC END ---

                # 4. Final Text Check & Write
                if should_process and is_text_file(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            
                            # Write a clear header for the AI to parse
                            outfile.write(f"\n{'='*50}\n")
                            outfile.write(f"FILE: {rel_path}\n")
                            outfile.write(f"{'='*50}\n\n")
                            outfile.write(content)
                            outfile.write("\n")
                            
                            print(f"Added: {rel_path}")
                            file_count += 1
                    except Exception as e:
                        print(f"Skipping {rel_path} (Error reading file: {e})")

    print(f"\n--- DONE ---")
    print(f"Processed {file_count} files.")
    print(f"Output saved to: {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    main()