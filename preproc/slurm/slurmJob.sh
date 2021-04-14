## Contributors: Matt Ware
#!/bin/bash
source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh

# Basepath to experiments will not change
basepath=/cds/data/psdm/tmo

# Read command line input
# The structure below reads in input of the type:
# ./slurmJob.sh --nodes=1 --dir=.
# This parallels the input to preproc.py
while [ $# -gt 0 ]; do
  case "$1" in
    --cores=*)
      cores="${1#*=}"
      ;;
    --directory=*)
      directory="${1#*=}"
      ;;
    --python=*)
      python="${1#*=}"
      ;;
    --queue=*)
      queue="${1#*=}"
      ;;
    --exp=*)
      exp="${1#*=}"
      ;;
    --run=*)
      run="${1#*=}"
      ;;
    --nevent=*)
      nevent="${1#*=}"
      ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument.*\n"
      printf "***************************\n"
      exit 1
  esac
  shift
done

# Print input for user to review
echo Running on $cores
echo Saving to $directory
echo Submitting to $queue
echo Using python script $python
echo Processing experiment $exp
echo Analyzing run $run
echo Processing nevents $nevent

# Print directory generation dialog
echo Checking if directory structure exists. If not, generating.
if [ ! -d "$basepath/$exp" ]; then
  echo Experiment does not exist. Exiting.
  exit
fi

newdir=$basepath/$exp/scratch/$USER
if [ ! -d "$newdir" ]; then
  echo User directory does not exist. Generating $newdir
  mkdir $newdir
  sleep 1
fi

newdir=$basepath/$exp/scratch/$USER/slurm
if [ ! -d "$newdir" ]; then
  echo SLURM directory does not exist. Generating $newdir
  mkdir $newdir
  sleep 1
fi

newdir=$basepath/$exp/scratch/$USER/slurm/$directory
if [ ! -d "$newdir" ]; then
  echo $directory directory does not exist. Generating $newdir
  mkdir $newdir
  sleep 1
fi

# Submit the batch job with the correct inputs
sbatch -p $queue --ntasks $cores --ntasks-per-node 16 \
         --output=$newdir/$run.out \
         --error=$newdir/$run.error \
         --wrap=\
"mpirun python -u $python --cores=$cores --directory=$newdir --run=$run --exp=$exp --nevent=$nevent"