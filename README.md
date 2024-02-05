# General
Repository for the trianing for: Hardware-Aware Quantization for Multiplierless
Neural Network Controllers

# Installation
Packages you need

`pip install tqdm
pip install pydot
pip install numpy==1.21.6
pip install tensorflow==2.10.0
pip install matplotlib
pip install scipy==1.10.1
pip install onnx==1.13.1´

- run the create_coeff.py
- run the create_scripts.py in the expiriments/ folder (Choose old or new method there. The old one is the original from the paper. Later I found out that with more optimization of the code and by removing memory leaks results get better with a much much bigger batch size...)
- run the all_experiments.sh in the expiriments/ folder
- run the create_plots.py to see results
