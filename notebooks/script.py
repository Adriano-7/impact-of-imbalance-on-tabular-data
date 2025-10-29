import subprocess
import os
import sys

def convert_notebook_to_script(notebook_path):
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook file not found at '{notebook_path}'")
        return None

    print(f"Attempting to convert '{notebook_path}'...")
    
    command = [
        sys.executable, 
        "-m", "jupyter",
        "nbconvert",
        "--to", "script",
        notebook_path
    ]
    
    try:
        subprocess.run(
            command,
            check=True,       
            capture_output=True, 
            text=True
        )
        
        output_path = os.path.splitext(notebook_path)[0] + '.py'
        
        if os.path.exists(output_path):
            return output_path
        else:
            print("Error: Conversion command ran, but the output file was not found.")
            return None
            
    except FileNotFoundError:
        print("Error: 'jupyter' command not found.")
        print("Please ensure Jupyter is installed (`pip install jupyter`) and in your system's PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print("An error occurred during notebook conversion.")
        print(e.stderr)
        return None

def main():
    notebook_files = [f for f in os.listdir('.') if f.endswith('.ipynb')]

    if not notebook_files:
        print("No Jupyter notebooks (.ipynb files) found in the current directory.")
        return

    print(f"Found {len(notebook_files)} notebook(s) to convert.\n")
    
    successful_conversions = 0
    failed_conversions = []
    
    for notebook in notebook_files:
        python_script_path = convert_notebook_to_script(notebook)
        
        if python_script_path:
            successful_conversions += 1
            print(f" Converted to '{python_script_path}'")
        else:
            failed_conversions.append(notebook)
            print(f"Could not convert '{notebook}'")

    print("\n Batch Conversion Complete ")
    print(f"Total Notebooks Found: {len(notebook_files)}")
    print(f"Successful Conversions: {successful_conversions}")
    
    if failed_conversions:
        print(f"Failed Conversions: {len(failed_conversions)}")
        print("The following notebooks failed to convert:")
        for failed_file in failed_conversions:
            print(f"  - {failed_file}")
    
    print("")


if __name__ == "__main__":
    main()