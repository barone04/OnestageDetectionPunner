## Extend from [PruneFishNet](https://github.com/barone04/PrunedFishNet.git). Prune one-stage detection models

### NORTON comparison path

`norton.py` applies a NORTON-style CP decomposition inside the same YOLOv1
training/evaluation environment. It keeps the detector, dataset, loss,
finetuning budget, and benchmark scripts shared with the channel-pruning
pipeline, but reports NORTON as a separate decomposition-based variant because
it changes topology instead of removing channels.

Example:

```bash
bash scripts/run_norton.sh
```
