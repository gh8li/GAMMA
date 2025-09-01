# GAMMA

> This repository contains the code of GAMMA*, our implementation of a GPU-based continuous subgraph matching algorithm described in the ICDE'2024 paper "GPU-Accelerated Batch-Dynamic Subgraph Matching" By Qiu et.al. [[link](http://doi.org/10.1109/ICDE60146.2024.00248)].


## Directory Structure

The main directories and files in this project are organized as follows:

```
Gamma/
├── build/           # Compiled binaries and build files
├── datasets/
├── graph/           # Data structures
├── kernels/         # CUDA kernels for graph update and matching
├── utils/           # Utility code
├── gamma.h          
├── main.cpp         
├── CMakeLists.txt   # CMake build configuration
├── README.md        # Project documentation
```


## Compile

This program requires cmake (Version >= 3.10), Make (Version >= 3.82), GCC (Version >= 11.2.0), and nvcc (Version >= 11.1). You can compile the code by executing the following commands from the root directory of the project. 

```shell
mkdir build
cd build
cmake ..
make
cd ..
```

## Execute

After compilation, the binary file will be in the `build/` directory. You can execute GAMMA* using the following command.

```shell
./build/gamma --query <query-graph-path> --data <data-graph-path>  --update <update-path>
```

### Commandline Parameters
The commandline parameters are listed in the following table.

| Parameters  | Description                               | Valid Value     | Default Value |
|-------------|-------------------------------------------|-----------------|---------------|
| --query     | Path to the query graph file.             | n/a             | n/a           |
| --data      | Path to the data graph file.              | n/a             | n/a           |
| --update    | Path to the update file.                  | n/a             | n/a           |
| --batch_size| Number of update edges in a batch.        | 1-4294967295    | 4294967295 (all in one batch)            |
| --show_gpu_memory| Print GPU memory usage or not.       | True/False      | False            |
| --print_indexing_time| Print indexing time or not.      | True/False      | False            |
| --device    | GPU ID for execution.                     | 0-7             | 0             |

## Input File Format
Both the input query graph and data graph are vertex-labeled and edge-labeled. Each vertex is represented by a distinct unsigned integer (from 0 to 4294967295). There is at most one edge between two arbitrary vertices. A vertex and an edge are formatted as `v <vertex-id> <vertex-label>` and `e <vertex-id-1> <vertex-id-2> <edge-label>`, respectively. The two endpoints of an edge must appear before the edge. For example, 

```
v 0 3
v 1 0
v 2 6
v 3 1
v 4 5
v 5 1
v 6 0
v 7 3
e 0 1 0
e 0 2 0
e 0 5 0
e 1 3 0
e 2 3 0
e 2 5 0
e 3 4 0
e 3 6 0
e 6 7 0
```

The update file contains only update edges, represented in the same format as in the data graph.


## Datasets and Querysets

The graph datasets and their corresponding querysets used in our paper can be downloaded [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/xsunax_connect_ust_hk/Et-cxVY7l5FCoZoKeDyMzmQBaCBn8ffbPFFQfIFOqGIodA?e=4vT3OI). The dataset zip files should be uncompressed inside the `datasets/` directory.
