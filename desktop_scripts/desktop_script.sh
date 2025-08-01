if [ $# -ne 6 ]; then
        echo "Usage: $0 <power_profiling_root> \
                        <power_file_name>      \
                        <latency_file_name>    \
                        <power_mode (0,1)      \
                        <inference_time>       \
                        <username@IP_addr>     "                     
        exit 1
fi

root_dir=$(realpath $1)
onnx_dir=$root_dir"/data/dws_2/onnx/"
trt_output_dir=$root_dir"/data/dws_2/trt/"
csv_output_dir=$root_dir"/data/dws_2/power_latency_output/"
txt_output_dir=$root_dir"/data/dws_2/tegrastat_output/"
power_fname=$2
latency_fname=$3
power_mode=$4
inference_time=$5
host_addr=$6

jetson_addr="user@ip_addr"
ssh_work_dir="~/user/Desktop/tmp_ssh_files/"

cd $root_dir

for onnx_fname in "$onnx_dir"/*.onnx; do
    fname=$(basename "$onnx_fname" ".onnx")

    echo "Loading $onnx_fname onto Jetson Nano"
    scp $onnx_fname $jetson_addr:$ssh_work_dir

    echo "Working on $onnx_fname"
    echo "SSH into Jetson Nano"

    ssh $jetson_addr "cd $ssh_work_dir;                      \
                      export power_mode=$power_mode;         \
                      export inference_time=$inference_time; \
                      export host_addr=$host_addr;           \
                      export trt_output_dir=$trt_output_dir; \
                      export txt_output_dir=$txt_output_dir; \
                      export csv_output_dir=$csv_output_dir; \
                      ../script/jetson_script.sh;"

    python3 ./processing_data/power_and_latency_grouper.py $csv_output_dir $power_fname $latency_fname

    rm $onnx_fname
    echo "Completed $onnx_fname"
    echo "_____________________"
done