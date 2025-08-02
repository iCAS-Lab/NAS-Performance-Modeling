# Accelerating Hardware-Aware NAS with ML-Based
This is the GitHub repository for the paper **Accelerating Hardware-Aware NAS with ML-Based Edge GPU Performance Modeling**, presented at the IEEE International Conference on Omni-layer Intelligent Systems in August 2025. The paper can be accessed at the following link: [TBD]

# Motivation
In recent years, attention has increasingly turned to practical deployment, especially on edge devices where computational resources, energy, and latency are tightly constrained. To accommodate this, methods such as hardware-in-the-loop NAS have been used to optimize a given model architecture for a given device. Unfortunately, a major limitation of these conventional approaches lies in their computational cost. Optimizing for multiple objectives such as accuracy and energy consumption is very resource intensive and time consuming, especially when evaluation is done on physical edge hardware. To overcome this bottleneck, we propose a predictive-model assisted NAS framework that significantly accelerates the search process by eliminating the need for real-time evaluation. By replacing the physical edge device with a predictive model capable of predicting the latency and energy consumption of a given model, this framework drastically speeds up the NAS process, proposing a valid alternative to hardware-in-the-loop NAS.
<p align="center">
<img width="928" height="341" alt="Screenshot 2025-08-02 151002" src="https://github.com/user-attachments/assets/cd476abe-14c1-4884-988f-ba3c6ed3949e" />
</p>

# Results
<p align="center">
<img width="1026" height="321" alt="image" src="https://github.com/user-attachments/assets/1e8fe75c-b53d-4e27-8dac-28f142d1e127" />
</p>
Our predictive model was able to achieve a MAPE of 15.28% for energy prediction and 15.50% for latency prediction on a selection of real world models. While these numbers are not world class, they highlight the pheasibility of our method. The predictive model relied heavily on the percentage of MACs that corresponded to a given kernel size. With this in mind, given that the model seemed to perform much worse on "efficient" CNN architectures such as ShuffleNet, we suspect that a larger dataset corresponding to a broader set of MAC distributions is required to achieve coverage of all SOTA models. <br>

<p align="center">
   <br>
<img width="582" height="331" alt="image" src="https://github.com/user-attachments/assets/e298d586-4f58-4706-b47c-f4d1616ee3a2" />
</p>
Results of our NAS framework showed a vast drop in runtime when compared to the hardware-in-the-loop approach (hours vs days). While accuracy and energy consumption numbers were not quite at the level of SOTA models, this experiment serves as a proof of concept, leaving the door open for future investigations. 

# Dataset
To train the predictive model, a dataset of arbitrary model architectures was created and subsequently ran on the Jetson Nano. Out of this process, two different datasets were created which are available for public use. Firstly, the primary dataset of energy and latency measurements from 12,000+ CNN models run on the Jetson Nano is available in the CSV format. Secondly, these 12,000 models are available in the TensorRT format for a replication or follow-up to our investigation.

- [Jetson Nano energy and latency dataset](data/dataset.csv)

- The link to the TRT dataset will be available shortly.

# Repository Structure
Below is a basic overview of the structure of the repository

1. `analysis` Folder
   - Contains scripts for various analyses and the methods used to develop predictive models.
  
4. `data` Folder
   - Stores all related data.
   - Organized into different files because data collection was done in stages:
     - Initially collected data for convolutional models with layers 1-5.
     - Followed by collection data for convolutional models with layers 6-10.
     - Due to a lack of high MAC models, further data collection was made for high MAC models.
     - Next, to help with architectures like MobileNet, data was collected for depthwise-separable (DWS) models.
     - Finally, data was collected for various combinations of mixed kernel models in hopes of further generalizing our dataset and predictor.

2. `desktop_scripts` Folder
   - Includes all scripts that are used to automate data collection from a host machine.

3. `jetson_scripts` Folder
   - Includes all scripts that run on the Jetson device.
   - Contains scripts for converting ONNX models to TensorRT, as well as scripts for measuring power consumption and latency.

5. `model_generation` Folder
   - Contains scripts for creating new models.

6. `Models` Folder
   - Contains the gradient boosting and random forest based predictive models as well as the best model as a result of the predictive model based NAS.
  
7. `NAS` Folder
   - Includes scripts used to conduct the predictive model based NAS.

7. `processing_data` Folder
   - Includes scripts used to process the collected data.
   - Handles data collected on the Jetson Nano, which is stored in `.txt` or`.csv` files.

8. `layer_mac_param_gen` File
   - When generating models for data collection, we had not yet identified the "MAC percentage" of different kernels as an important factor within the predictive model. Consequently, this file was needed to retroactively determine the MAC percentage of each kernel size for all the previously generated models.

# Contributors
If you have any questions regarding the paper, code, or dataset, please contact us on LinkedIn.

**Paper Authors:** <br>
Aishneet Juneja - University of South Carolina <br>
Matthew Grenier - University of South Carolina <br>
Md Hasibul Amin - University of South Carolina <br>
Ramtin Zand - University of South Carolina

# Citation
TBD
