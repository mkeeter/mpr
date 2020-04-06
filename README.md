# libfive-cuda
This is the reference implementation for
"Massively Parallel Rendering of Complex Closed-Form Implicit Surfaces",
a technical paper which will be presented at SIGGRAPH 2020.

## Reproducing our results on AWS
You can reproduce the results in the paper for about $5 on AWS!

First, make sure that your AWS account has permission to create a `p3.2xlarge` instance.
New accounts don't, so I had to email support and ask for an increase in my vCPU limits.

Keep in mind, this instance costs $3.06/hour,
so you'll want to be very careful about **turning it off when not benchmarking**;
leaving it on for a month will cost you a cool **$2276**.

Once you've gotten permission to create the instance,
spin up an server with the latest version of
`Deep Learning Base AMI (Ubuntu 18.04)`.
I used `Version 21.0 (ami-0b98d7f73c7d1bb71)`, but you should use the most recent release.

SSH into the server and run a bunch of commands:
```
# Install dependencies
sudo apt install mesa-common-dev ninja-build

# Install a recent version of eigen
wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
tar -zxvf eigen-3.3.7.tar.gz
cd eigen-3.3.7
mkdir build
cd build
cmake ..
sudo make install

# Install libfive-cuda
cd
git clone git@github.com:mkeeter/libfive-cuda
cd libfive-cuda
git submodule update --init --recursive
mkdir build
cd build
cmake -GNinja -DBIG_SERVER=ON ..
ninja
```

At this point,
you can reproduce the benchmarks in the paper by running `../run_benchmarks.sh`
(from the `build` directory).
This will print a bunch of performance values, starting with

```
============================================================
                      2D benchmarks
============================================================
Text benchmark
256 5.29331 0.261052
512 4.21138 0.00523862
1024 3.85596 0.00625019
...
```
The three columns are size,
frame time (in milliseconds),
and standard deviation.

The benchmarking script will save the output images
into a subfolder for each model:
```
prospero
gears_2d
architecture
gears_3d
bear
```

Remember to **turn off the server when you're done**.

## Building on MacOS
Install [Homebrew](https://brew.sh) and [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html),
then run something like
```
brew install cmake pkg-config eigen libpng qt guile boost ninja
git clone git@github.com:mkeeter/libfive-cuda
cd libfive-cuda
git submodule update --init --recursive
mkdir build
cd build
env CUDACXX=/usr/local/cuda/bin/nvcc cmake -GNinja ..
ninja
```

## Disclaimer
This is research code maintained in my spare time,
without institutional or commercial backing.
I don't have the resources to support every Linux distro,
though I'll provide best-effort support
for running on AWS in the configuration described above.
If you're having trouble on your particular Linux distro,
please debug it independently and open a PR rather than an issue.
