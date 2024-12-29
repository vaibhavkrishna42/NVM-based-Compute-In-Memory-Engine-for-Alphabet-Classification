### `Devices and Technologies for AI and Neuromorphic Computing (EE6347)`  
### `Guide : Professor Bhaswar Chakrabarti, IIT Madras`  

This project's goal is to design the whole stack: generate the dataset, train a simple Multi-Layer Perceptron (MLP) model to classify alphabets correctly and finally build a hardware NVM-based model to perform the inference.  

### `Dataset Generation`  
The primary reference used for the same : [B Chakrabarti et al. (2018)](https://www.nature.com/articles/s41467-018-04482-4)  
 A set of ideal images were taken from the above paper. The training set consisted of images with 1 randomly flipped pixel whereas the testing set had images with 2 randomly flipped images.  

### `Hardware-Aware Training`  
It was important to account for the binarized weights that were to be loaded onto the RRAM array. 
To minimize the quantization error, the weights were pushed towards the extremes i.e. 0 and 1 with the help of a regularization term during training to ensure that inference loss wasn't significant.
We achieved a software accuracy of **71%**.  

### `Hardware Inference`  
A 2-layer RRAM array was designed for hardware inference. The Multiply-Accumulate (MAC) value was sent through a comparator to mimic a squeezing activation function. 
We noticed a surprisingly similar accuracy with and without an activation in the software sicne the output of layer 1 had to be further binarized before being fed into layer 2.
The hardware inference yielded an accuracy of **58%**.  

### A detailed report of the project can be found at : [PDF](EE6347_Project_Report.pdf)
