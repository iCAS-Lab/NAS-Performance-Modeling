# Aim
The project is divided into two phases. First, the goal is to develop a predictive model that can estimate the power consumption and latency of a given model architecture during inference on the Jetson Nano. In the second phase, this predictive model will be used to perform Neural Architecture Search (NAS) and compare the results with NAS conducted directly on the Jetson Nano.

# Structure of Repository
This is not the final structure, as I am still in the process of organizing it. However, to simplify things, I have segregated most components as follows:

1. `analysis` Folder
   - Contains all analyses performed on the collected data.
   - Inside the `code` subfolder, you will find scripts for various analyses and the methods used to develop predictive models.

2. `jetson_scripts` Folder
   - Includes all scripts that run on the Jetson device.
   - Contains scripts for converting ONNX models to TensorRT, as well as scripts for measuring power consumption and latency.

3. `data` Folder
   - Stores all related data.
   - Organized into different files because data collection was done in stages:
     - Initially collected data for fully connected networks.
     - Followed by convolutional models with layers 1-5.
     - Finally, convolutional models with layers 6-10.

4. `model_generation` Folder
   - Contains scripts for creating new models.(will be adding more scripts)

5. `processing_data` Folder
   - Includes scripts used to process the collected data.
   - Handles data collected on the Jetson Nano, which is stored in `.txt` or`.csv` files.
