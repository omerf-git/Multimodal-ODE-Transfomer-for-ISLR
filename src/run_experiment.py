import subprocess
import sys
import re
from pathlib import Path

def parse_config(config_path: Path) -> dict:
    """
    Parses a simple .sh configuration file and returns a dictionary.
    Supports 'KEY="VALUE"' or 'KEY=VALUE' format.
    """
    if not config_path.is_file():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    config = {}
    # Regex to find variable assignments: KEY=VALUE or KEY="VALUE"
    pattern = re.compile(r'^\s*([\w_]+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s#]+))')

    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            match = pattern.match(line)
            if match:
                key = match.group(1)
                # Find matched value (with or without quotes)
                value = next((g for g in match.groups()[1:] if g is not None), None)
                config[key] = value
    return config

def main():
    """
    Reads configuration and runs the training script.
    """
    # Get script directory
    base_dir = Path(__file__).parent
    config_path = base_dir / 'config.sh'

    # 1. Parse configuration
    config = parse_config(config_path)

    # 2. Build command line arguments
    # Base command (assuming run from src directory)
    command = ['python', '-m', 'train']

    # Add arguments for each key/value pair in config
    for key, value in config.items():
        # Special case handling for NORM_FIRST
        if key == 'NORM_FIRST':
            # Add --no-norm-first flag if value is 'False'
            if str(value).lower() == 'false':
                command.append('--no-norm-first')
            # Do nothing if 'True' (default behavior)
            continue

        # Add all other parameters
        arg_name = f'--{key.lower()}'
        command.append(arg_name)
        command.append(str(value))

    # Print constructed command
    print("Executed Command:")
    # Print joined command for readability
    print(' '.join(command))
    print("-----------------------------------------------------")

    # 3. Run command
    try:
        # subprocess.run waits until command completes
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during training. Exit code: {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("Error: 'python' command not found. Please ensure Python is installed and in your PATH.")
        sys.exit(1)

if __name__ == '__main__':
    main()