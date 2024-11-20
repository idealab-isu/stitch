# STITCH
**S**urface reconstruc**T**ion using **I**mplicit neural representations with **T**opology **C**onstraints and persistent **H**omology



## Setting up the repository
```bash
1. conda env create -f tda.yml
2. Make a directory called 'data' in the root directory of the repository
3. Place your point cloud data in the 'data' directory
4. Edit the 'cubicalLayer.conf' file to set the correct paths for the data, base directory, etc.
```

## Training
```bash
python run/runCubicalLayer.py --gpu YOUR_GPU_INDEX_GOES_HERE --conf confs/cubicalLayer.conf --mode curriculum --dataname NAME_OF_INPUT_FILE_GOES_HERE --dir NAME_OF_SAVING_DIRECTORY_GOES_HERE
```
## Inference
```bash
python run/runCubicalLayer.py --gpu YOUR_GPU_INDEX_GOES_HERE --conf confs/cubicalLayer.conf --mode validate_mesh --dataname NAME_OF_INPUT_FILE_GOES_HERE --dir NAME_OF_SAVING_DIRECTORY_GOES_HERE

# If inferring for the same experiment as training then,
# use the same --dataname and --dir arguments as training

```
