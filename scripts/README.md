# How To Run Benchs
## 1. main experiment
1. Set `TUNING_BUDGET` and `SEED` at `src.constant`
2. Manipulate configs of `run_src.py` as you want
3. run `run_src.py`
4. run `results/n_seed_to_0.sh` with arg `{SEED}` to move created result into base result directory
5. In order to postprocess, run `results/postprocess.py` after manipulating its configs
6. Done. Check the results