# This script must be run through the deskop_script.sh script from
# the host computer. deskop_script.sh will transfer the required .onnx
# file and set the following variables:
#
# host_addr: username@IP_addr of host computer
#
# trt_output_dir: location within host where trt files should be saved
#
# csv_output_dir: location within host where csv files from Power.py
# and latency.py should be saved
#
# txt_output_dir: location within host where txt files from Power.py
# and latency.py should be saved

onnx_fname=$(ls | grep .onnx)
fname=$(basename "$onnx_fname" ".onnx")

echo "Converting $onnx_fname to .trt"
trt_fname=$fname".trt"
python3 ../script/newonnx2trt.py . .

output_file="stats_${fname}_mode${power_mode}_${inference_time}.txt"

echo "Changing power mode to $power_mode..."
echo "Saving stats to $output_file"

sudo nvpmodel -m $power_mode

python3 ../script/Power.py $trt_fname $inference_time $output_file

cur_dir=$(pwd)
python3 ../script/latency.py $cur_dir

scp $trt_fname $host_addr:$trt_output_dir
scp *.csv $host_addr:$csv_output_dir
scp *.txt $host_addr:$txt_output_dir

rm *.csv *.txt *.onnx *.trt

exit