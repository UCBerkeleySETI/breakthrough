# Neural Network for Fast Radio Burst Detection

This is an inference script for detecting FRB with a neural network. It is designed for use with the Breakthrough Listen 
data product at the Green Bank Telescope. 

Software Requirements:
```
CUDA9.0 
tensorflow > 1.4.0
sigpyproc
```

## Sample Usage
```
python inference_gbtc.py --model ./models/GBTC.pb --filterband_dir /path/to/data/ --threshold 0.99
```

### Researh Code
research_code directory contains the main code used during the research. The code here are messy and not intended for general use. 
