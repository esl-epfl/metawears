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
gcc -O3 -o epilepsy_detector.out main.c epilepsyTransformer.c transformer_C/*.c -I. -ISYLT-FFT -Itransformer_C -lm -lstdc++
```
- -O3: Enables high-level compiler optimizations.
- -lstdc++: Links the standard C++ library, which is needed because the data files have a .cpp extension.

After successful compilation, you can run the program directly from your terminal:
```bash
 ./epilepsy_detector.out
```

When you run the program, you should see the following output, including the performance metrics and the classification result. The exact timing values may vary slightly depending on your machine.

```
Starting STFT preprocessing...
STFT finished.
Starting transformer inference...
Inference finished.
Starting prototype distance calculation...
Prototype calculation finished.

--- Performance Metrics ---
STFT Preprocessing Time: 
Transformer Inference Time: 
Prototype Calculation Time:
---------------------------
Total Application Time:
---------------------------
Distances :
From the prototype of class 0 = 462
From the prototype of class 1 = 107749
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

The expected output:
```
start forward_mobilenet inference ........
############################Bottleneck#############################
############################Bottleneck#############################
############################Bottleneck#############################
############################Bottleneck#############################
############################Bottleneck#############################
############################Bottleneck#############################
############################Bottleneck#############################
############################Bottleneck#############################
############################Bottleneck#############################
############################Bottleneck#############################
############################AVG Pooling############################
Data [0] = 8.98
Data [1] = 14.20
Data [2] = 13.22
Data [3] = 0.00
Data [4] = 0.00
Data [5] = 18.24
Data [6] = 0.00
Data [7] = 0.00
Data [8] = 0.00
Data [9] = 0.00
Data [10] = 12.41
Data [11] = 0.00
Data [12] = 0.00
Data [13] = 9.65
Data [14] = 13.47
Data [15] = 0.00
Data [16] = 0.00
Data [17] = 0.00
Data [18] = 0.00
Data [19] = 14.24

--- MobileNet Performance Breakdown ---
Stage 1 (Initial Conv) Time: 
Stage 2 (Bottlenecks) Time: 
Stage 3 (Classifier) Time:   
---------------------------------------
Total Inference Time:      
---------------------------------------
-----------------------------End forward---------------------------------
```

