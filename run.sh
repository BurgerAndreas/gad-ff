source .venv/bin/activate


# uv run scripts/eval.py --max_samples=1000 --ckpt_path=ckpt/alpha.ckpt --redo=True;
# uv run scripts/eval.py --max_samples=1000 --ckpt_path=ckpt/left.ckpt --redo=True;  
# uv run scripts/eval.py --max_samples=1000 --ckpt_path=ckpt/left-df.ckpt --redo=True; 
# uv run scripts/eval.py --max_samples=1000 --ckpt_path=ckpt/eqv2.ckpt --redo=True;

# uv run scripts/eval.py --max_samples=1000 --ckpt_path=ckpt/alpha_orig.ckpt --redo=True;
# uv run scripts/eval.py --max_samples=1000 --ckpt_path=ckpt/left_orig.ckpt --redo=True;  
# uv run scripts/eval.py --max_samples=1000 --ckpt_path=ckpt/left-df_orig.ckpt --redo=True;
# uv run scripts/eval.py --max_samples=1000 --ckpt_path=ckpt/eqv2_orig.ckpt --redo=True;

# uv run scripts/eval.py --max_samples=1000 --ckpt_path=ckpt/hesspred_v2.ckpt --hessian_method=predict --redo=True;

##################### RGD1.lmdb #####################

uv run python scripts/eval.py -c ckpt/eqv2.ckpt -d RGD1.lmdb -m 1000 -r True -hm autograd;
uv run python scripts/eval.py -c ckpt/eqv2_orig.ckpt -d RGD1.lmdb -m 1000 -r True -hm autograd; 
uv run python scripts/eval.py -c ckpt/hesspred_v2.ckpt -d RGD1.lmdb -m 1000 -r True -hm predict;

uv run python scripts/eval.py -c ckpt/eqv2.ckpt -d RGD1.lmdb -r True -hm autograd;
uv run python scripts/eval.py -c ckpt/eqv2_orig.ckpt -d RGD1.lmdb -r True -hm autograd; 
uv run python scripts/eval.py -c ckpt/hesspred_v2.ckpt -d RGD1.lmdb -r True -hm predict;

# # other models for eigval error etc
# uv run python scripts/eval.py -c ckpt/left.ckpt -d RGD1.lmdb -m 1000 -r True -hm autograd; 

# uv run python scripts/eval.py -c ckpt/left-df.ckpt -d RGD1.lmdb -m 1000 -r True -hm autograd; 

# uv run python scripts/eval.py -c ckpt/alpha.ckpt -d RGD1.lmdb -m 1000 -r True -hm autograd; 


# hip
# uv run python scripts/eval.py -c ckpt/hesspred_v1.ckpt -d RGD1.lmdb -m 1000 -r True -hm predict;

# uv run python scripts/eval.py -c ckpt/hesspred_v2.ckpt -d RGD1.lmdb -m 1000 -r True -hm predict;
