# Accelerating Hardware-Aware NAS with ML-Based
Edge GPU Performance Modeling
The project is divided into two phases. First, the goal is to develop a predictive model that can estimate the energy consumption and latency of a given model architecture during inference on the Jetson Nano. In the second phase, this predictive model will be used to perform Neural Architecture Search (NAS) and compare the results with NAS conducted directly on the Jetson Nano.

# Structure of Repository
Below is a basic overview of the structure of the repository

1. `analysis` Folder
   - Contains scripts for various analyses and the methods used to develop predictive models.

2. `desktop_scripts` Folder
   - Includes all scripts that are used to automate data collection from a host machine.

3. `jetson_scripts` Folder
   - Includes all scripts that run on the Jetson device.
   - Contains scripts for converting ONNX models to TensorRT, as well as scripts for measuring power consumption and latency.

4. `data` Folder
   - Stores all related data.
   - Organized into different files because data collection was done in stages:
     - Initially collected data for convolutional models with layers 1-5.
     - Followed by collection data for convolutional models with layers 6-10.
     - Due to a lacking of high MAC models, further data collection was made for high MAC models.
     - Next, to help with architectures like MobileNet, data was collected for depthwise-seperable (DWS) models.
     - Finally, data was collected for various combinations of mixed kernel models in hopes of further generalizing our dataset and predictor.

5. `model_generation` Folder
   - Contains scripts for creating new models.(will be adding more scripts)

6. `processing_data` Folder
   - Includes scripts used to process the collected data.
   - Handles data collected on the Jetson Nano, which is stored in `.txt` or`.csv` files.

7. `processing_data` Folder
   - Includes scripts used to process the collected data.
   - Handles data collected on the Jetson Nano, which is stored in `.txt` or`.csv` files.

7. `layer_mac_param_gen` File
   - When generating models for data collection, we had not yet identified the "MAC percentage" of different kernels as a important factor within the would be predictive model. Consequently, this file was needed to retroactively determine the MAC percentage of each kernel size for all the previously generated models.