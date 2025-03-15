# General
Repository for the trianing for: Hardware-Aware Quantization for Multiplierless
Neural Network Controllers (https://ieeexplore.ieee.org/document/10090271) (https://hal.science/hal-03827660/) (

T. Habermann, J. Kühle, M. Kumm and A. Volkova, "Hardware-Aware Quantization for Multiplierless Neural Network Controllers," 2022 IEEE Asia Pacific Conference on Circuits and Systems (APCCAS), Shenzhen, China, 2022, pp. 541-545, doi: 10.1109/APCCAS55924.2022.10090271.
keywords: {Training;Quantization (signal);VHDL;Embedded systems;Throughput;Hardware;Safety;neural network controllers;machine learning;hardware-aware quantization;quantization-aware training}, 

# Installation
Packages you need
```
pip install tqdm
pip install pydot
pip install numpy==1.21.6
pip install tensorflow==2.10.0
pip install matplotlib
pip install scipy==1.10.1
pip install onnx==1.13.1
```
- run the create_coeff.py
- run the create_scripts.py in the expiriments/ folder (Choose old or new method there. The old one is the original from the paper. Later I found out that with more optimization of the code and by removing memory leaks results get better with a much much bigger batch size...)
- run the all_experiments.sh in the expiriments/ folder
- run the create_plots.py to see results
