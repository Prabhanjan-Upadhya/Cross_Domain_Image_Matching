module load cuda-toolkit/5.5.22
rm -f project
echo "Compiling Code"
nvcc -O2 -arch=sm_35 -o project project.cu
if [ "$?" != "0" ]; then
    echo "[Error] Compile Failed" 1>&2
    exit 1
fi
echo "Executing Code"
echo "-------------"
./project $1 $2
