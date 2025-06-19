import os


def find_project_root(
    start_path=None, markers=("pyproject.toml", ".git")
    ):
    """Walk up from start_path to find a directory containing one of the marker files."""
    if start_path is None:
        start_path = os.path.abspath(__file__)
    dir_path = os.path.dirname(start_path)
    while True:
        for marker in markers:
            if os.path.exists(os.path.join(dir_path, marker)):
                return dir_path
        parent = os.path.dirname(dir_path)
        if parent == dir_path:
            raise RuntimeError(f"Project root not found (looked for {markers})")
        dir_path = parent


if __name__ == "__main__":
    print(find_project_root())