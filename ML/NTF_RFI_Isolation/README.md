# Non Negative Tensor/Matrix Factorization - SETI
## Try It Out
You can try out the example note book that gives a good walkthrough on the implementation! [link](https://github.com/PetchMa/NTF_SETI_RFI/blob/master/Example_Non_Negative_Matrix_Factor.ipynb "link")

## How It Works
RFI is interference found that isn't localized in the sky but instead came near by the sources. Thus the signal should appear on multiple scans. The goal is to take out these "reoccuring signals". However we need to make sure we are not taking out true transient signals localized on the sky. 

The algorithm works by factoring the first scan A and taking the factors and using it as the initalization for the second factoring of B. The reason we do so is because the next factoring step will effectively overwrite the features encoded in B and but wouldn't overwrite features that are similar both spatially and in morphology. Then we take the signal and we cross multiple the factors of B with the factors of A. This forces the activation of features that are present in both A and B but features in just A or just B won't be shown as much as it would activate the noise it has learned to represent. 



## Results
The results are quite interesting. Here I took a snippet of 2 cadences of these samples `GBT_57513_78094_HIP30112_mid.h5` and `GBT_57513_78437_HIP29212_mid.h5`. We then took a snipet in `1394mHz` range where there is known notch filter artifacts that we wish to remove. We then normalized the data between 0-1. Then we injected two beams with `setigen` with a bandwidth of `0.01mHz`.  Then we told the algorithm to delete the RFI/Artifacts and it did so pretty well! 

However be aware that this does distort the original signal quite a bit and isn't super reliable at the moment. 

![](https://github.com/PetchMa/NTF_SETI_RFI/blob/master/assets/results_3.png?raw=true)

Here's an even better example with a simulated example with noise and without noise. It preformed very well! 

![](https://github.com/PetchMa/NTF_SETI_RFI/blob/master/assets/results_1.png?raw=true)
