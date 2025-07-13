# Embedded Inference Models in C

## Overview

This repository contains "C" implementations of two neural network models designed for efficient inference on embedded systems:
- Epilepsy Transformer: A transformer-based model that analyzes EEG signals to detect seizure activity.
- MobileNet: A lightweight convolutional neural network.

The code is written in plain "C" and is optimized for resource-constrained devices, utilizing fixed-point arithmetic.

## Getting Started

These instructions explain how to compile and run the C code on a target embedded device.


### 1. Clone the Repository
Clone this repository and initialize its submodules. The SYLT-FFT library is included as a submodule and is required for compilation.

```bash
git clone --recurse-submodules https://github.com/alirezaamir/MetaWearS.git 
cd MetaWearS 
```

### 2. Compile for a Target Device
The following commands will compile the Epilepsy Transformer project for a target. You may need to adjust the flags for your specific device.

First, navigate to the project directory:
```bash 
cd hw-inference/epilepsy_C/
```

This single command compiles all necessary files and creates an executable to run on your local machine:

```bash
gcc -O3 -o epilepsy_detector.out main.c epilepsyTransformer.c transformer_C/*.c data_cpp/*.cpp -I. -ISYLT-FFT -Itransformer_C -lm -lstdc++
```
- -O3: Enables high-level compiler optimizations.
- -lstdc++: Links the standard C++ library, which is needed because the data files have a .cpp extension.

After successful compilation, you can run the program directly from your terminal:
```bash
 ./epilepsy_detector.out
```


Also, this command compiles the MobileNet project to run on your local machine. Navigate to the `hw-inference/MobileNetAF/` directory and run:

```bash 
cd ../MobileNetAF
gcc -O3 -o mobilenet.out main.c weights.c mobilenet.c inference.c -lm
```
After successful compilation, you can run the program directly from your terminal:
```bash
 ./mobilenet.out
 ```

