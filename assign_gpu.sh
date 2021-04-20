# assign_gpu.sh

source /course/cs146/public/cs146-gpu-env/bin/activate

time python final.py -Tt -m xlmr -n 8 -lang related
