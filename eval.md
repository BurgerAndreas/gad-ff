

mkdir -p ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/eqv2.ckpt -O ckpt/eqv2.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/left-df.ckpt -O ckpt/left-df.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/left.ckpt -O ckpt/left.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/alpha.ckpt -O ckpt/alpha.ckpt

wget https://huggingface.co/yhong55/HORM/resolve/main/eqv2_orig.ckpt -O ckpt/eqv2_orig.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/left-df_orig.ckpt -O ckpt/left-df_orig.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/left_orig.ckpt -O ckpt/left_orig.ckpt
wget https://huggingface.co/yhong55/HORM/resolve/main/alpha_orig.ckpt -O ckpt/alpha_orig.ckpt

uv run scripts/size_eval.py -c ckpt/eqv2.ckpt
uv run scripts/size_eval.py -c ckpt/eqv2.ckpt -d geometries/dft_geometries.lmdb -hm predict
uv run scripts/size_eval.py -c ckpt/eqv2.ckpt -m 50 --redo