module load anaconda/python3/2022.05 mkl/64/2022/0/0
filename=$1.py
if [[ -f "$filename" ]]; then
    echo "$filename exists!"
else
    cp energy_disorder.py $filename
    python3 $filename > $1.out &
fi
