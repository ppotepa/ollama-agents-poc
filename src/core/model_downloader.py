import subprocess
import sys
from typing import Optional

def pull_model(model_name: str, show_progress: bool = True, docker_container: str = "ollama") -> int:
    """
    Pull a model using `docker exec ollama ollama pull <model_name>` and display progress.
    Args:
        model_name: Name of the model to pull (e.g., 'deepseek-coder:6.7b-instruct-q4_K_M')
        show_progress: Whether to display the download progress in real time
        docker_container: Name of the ollama docker container
    Returns:
        Exit code from the pull command
    """
    cmd = ["docker", "exec", docker_container, "ollama", "pull", model_name]
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        bufsize=1,
        encoding='utf-8',
        errors='replace'  # Handle encoding errors gracefully
    )
    if show_progress:
        try:
            for line in process.stdout:
                print(line, end="")
        except KeyboardInterrupt:
            print("\nDownload interrupted by user.")
            process.terminate()
            return 1
    process.wait()
    return process.returncode

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python model_downloader.py <model_name>")
        sys.exit(1)
    model = sys.argv[1]
    exit_code = pull_model(model)
    sys.exit(exit_code)
