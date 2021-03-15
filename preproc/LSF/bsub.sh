#!/bin/bash
#source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh #this sets environment for psana2. note psana2 not compatible with psana(1)

ver=v0
base_path=/cds/home/m/mrware/Workspace/Workspace/2021-02-tmolw56/2021-02-preproc
script=$base_path/preproc.py
log=$base_path/logs/run$1_$ver.log
server='psnehq'


if [ -z "$2" ]
then
    nodes=16
else
    nodes=$2
fi

echo $log
echo $script

export PS_SRV_NODES=1
bsub -o $log -q psnehq -n $nodes mpirun python $script $1
