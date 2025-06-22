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


if __name__ == "__main__":
    print(find_project_root())