# DeepSeti - Breakthrough Listen Deep Learning Search Tool
This is a python implementation of DeepSeti - an algorithm designed to detect anomalies for radio telescope data open sourced by Breakthrough Listen. These python scripts facilitates the custom architecture and training loops required for the DeepSeti algorithm to preform a multichannel search for anomalies. Main objective is to develop software that increases the computational sensitivity and speed to search for 'unknown' anomalies.  **NOTE:** *Currently this code only works for **MID-RES filterbank and h5 files**. Developments made further will preforming tests on full res products from GBT.*

![alt text](https://github.com/PetchMa/DeepSeti/blob/master/assets/code_block1.png)

# Introduction

The purpose of this algorithm is to help detect anomalies within the GBT dataset from Breakthrough Listen. The code demonstrates a potential method in accelerating ML SETI in large unlabeled datasets. This approach is an extension from the original paper [https://arxiv.org/pdf/1901.04636.pdf] by looking into preforming the final classification on the encoded feature vector by taking a triplet loss between an anchor, and positive or negative samples.


# Deep Learning Architecture

What makes this algorithm unique is that it *injects* an encoder, thats been previously trained on a classification dataset, into an autoencoder trained through unsupervised techniques. This method relies on a inital small labeled dataset where it is intermediately trained a CNN-LSTM classifier then injected it into the CNN-LSTM Auto Encoder. 

**Rationale**: The goal is to force the feature selection from CNN's to search for those desired labels while the unsupervised method gives it the “freedom” to familiarize with "normal data" and detect novel anomalies beyond the small labeled dataset. Both the supervised and unsupervised models are executed together and model injections occur intermittently.

*Reference diagram below*

<p align="center"> 
    <img src="https://github.com/PetchMa/DeepSeti/blob/master/assets/image%20(3).png">
</p>

# Preliminary - Results 
From current tests done on new data, it was able to generalize to a variety of usecases. The image below shows how sensitive the algorithm is to small and weak beams across multiple channels. Despite never being trained on the sinusoid beam on the left [sample A], it was able to detect the anomaly out of the dataset. This shows promise in the intended use case of the algorithm. 

<p align="center"> 
    <img src="https://github.com/PetchMa/DeepSeti/blob/master/assets/image%20(4).png">
</p>

The example detection is made by taking the MSE / euclidean distance between the two vectors, an anchor vector and a unknown vector. These spikes within the data are seen as anomolies by the algorithm giving us these two detections. 

<p align="center"> 
    <img src="https://github.com/PetchMa/DeepSeti/blob/master/assets/image%20(5).png">
</p>

# Round 1 -  2 Terabytes Search @BreakthroughListen April 08 2020
With the first round execution, the algorithm searched through the first 2 terabytes worth of Breakthrough listen Data in search for signs of "intelligence". Over the 20 hour compute time,
this signal was perhaps the strangest amongst its finds! Further analysis needed. But promising search. 
<p align="center"> 
    <img src="https://github.com/PetchMa/DeepSeti/blob/master/round_1_2020-04-08/analysis.png">

</p>
If you are an astronomer and would like to want to see the results of the first round searches, checkout the folder titled first round! The complete csv is also avliable. 
<p align="center"> 
    <img src="https://github.com/PetchMa/DeepSeti/blob/master/assets/animation.gif">
</p>





# Round 2 -  4 Terabytes Search - Coming soon!
Updates coming soon!!

# How To Use The Algorithm 

Some features are still under construction, however you can test the current powers of this algorithm following this simple guide below. ** Note: This will require Blimpy and Setigen to opperate properly.** Install these requirements by running the following commands in the terminal in your python enviroment. 

```
pip3 install -r requirements.txt
```

After getting the following setup. Download a radio observation from the UC Berkeley SETI open database. [http://seti.berkeley.edu/opendata]. Or get a test sample by typing this command...
```
wget http://blpd13.ssl.berkeley.edu/dl/GBT_58402_66623_NGC5238_mid.h5
```
Following that all you need to do is clone the repository, and navigate into the folder where its cloned in.
```
git clone https://github.com/PetchMa/DeepSeti.git
```

Once you're within the cloned folder, copy the code block into a new python script. Fill in the mising directories, and you can train a model on your custom data. **Note: you can also load a pretrained model called *encoder_injection_model_cudda.h5* which has been trained on 500,000 radio samples. Keep in mind this requires CUDDA supported devices + drivers. Try the vanillia encoder_injection_model(1).h5 without Cudda support.**


```python
%tensorflow_version 1.x
import tensorflow
from DeepSeti import DeepSeti

DeepSeti = DeepSeti()
DeepSeti.load_model_function(model_location='/content/encoder_injected_model_CUDA_04-13-2020.h5')
DeepSeti.load_anchor(anchor_location='/content/GBT_58402_66967_HIP66130_mid.h5')
print("Model Loaded... Executing search loop")
search_return = DeepSeti.prediction(test_location='/content/'+file_download,  
                top_hits=1, target_name=file_download,
                output_folder='/content/drive/My Drive/Deeplearning/SETI/output_folder/',
                numpy_folder='/content/drive/My Drive/Deeplearning/SETI/numpy_output_folder/')   
```
This example will search for the most confident candidates and return saved images of these candidates. You can checkout an example notebook that loads queries the database EXAMPLE => [https://github.com/PetchMa/DeepSeti/blob/master/Examples/DeepSeti_Engine.ipynb]


# Next Steps
In the next bit, this project will be scaled up to TPU's to allow for training on larger datasets. Goal is to train the model on over 1 million radio samples! Follow updates on my twitter! [https://twitter.com/peterma02] Feel free to reach out to me by email: peterxiangyuanma@gmail.com if you have any questions or concerns. 


# Acknowledgements
Special thanks to the UC Berkeley SETI Research Team for the continual support. => list of their work here: [https://github.com/UCBerkeleySETI]
Also thanks to Breakthrough Listen and Green Bank Telescope for making the data openly accessible. More on them here: [https://breakthroughinitiatives.org/initiative/1]



