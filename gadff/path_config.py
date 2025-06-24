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

ROOT_DIR = find_project_root()

# HORM dataset
DATASET_DIR_HORM_EIGEN = os.path.expanduser(
    "~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/"
)
DATASET_FILES_HORM = [
    "ts1x-val.lmdb", # 50844 samples
    "ts1x_hess_train_big.lmdb", # 1725362 samples
    "RGD1.lmdb", # 60000 samples
]

DATA_PATH_HORM_SAMPLE = os.path.join(ROOT_DIR, "data/sample_100.lmdb")


CHECKPOINT_DIR = os.path.join(ROOT_DIR, "ckpt")
CHECKPOINT_PATH_EQUIFORMER_HORM = os.path.join(CHECKPOINT_DIR, "eqv2.ckpt")

#########################################################################################################


def remove_dir_recursively(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            # Remove all files in the directory before removing the directory itself
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    # Recursively remove subdirectories if any
                    import shutil
                    shutil.rmtree(file_path)
            os.rmdir(path)
        else:
            os.remove(path)
    # success if path does not exist anymore
    return not os.path.exists(path)

def _fix_dataset_path(_path):
    if os.path.exists(_path):
        # set absolute path
        return os.path.abspath(_path)
    else:
        new_path = os.path.join(DATASET_DIR_HORM_EIGEN, _path)
        if os.path.exists(new_path):
            return new_path
        else:
            raise FileNotFoundError(f"Dataset path {_path} not found")

if __name__ == "__main__":
    print(find_project_root())