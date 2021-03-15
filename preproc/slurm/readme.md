# TMO preprocessing
Contributors: Matt Ware, Elio Champenois, Taran Driver

## Required libraries
You'll need to source the newest psana2 libraries. Run `source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh` on the terminal.

## Mandatory updates


## Suggested updates


## Preprocessing flow
1) Automated job submission

	* Controlled by `autopreproc.py`, which calls `slurmJob.sh`.
	* Run via `python autopreproc.py --keyword-args=...`

2) Single job submission
	* Controlled by `preproc.py`

## File descriptions

### autopreproc.py
Automates the batch submission, calls slurmJob.sh.

Submit on command line like
```bash
python autopreproc.py --cores=32 --run=215 --runStop=216 --exp=tmolv2918 --queue=psfehq --nevent=10000 --directory=lw56-sample
```

### slurmJob.sh
Submits SLURM jobs. Try running first with psdebugq.

Submit on command line like
```bash
./slurmJob.sh --cores=256 --run=215 --exp=tmolv2918 --nevent=100000 --directory=test --python=preproc.py --queue=psfehq
```


### preproc.py
Run a single or multi-cored analysis of the data.

Run as
```bash
mpirun python -u preproc.py --run=215 --exp=tmolv2918 --nevent=100000 --directory=test
```

### cmdInput.py
Specifies the command line input structure that is read in by preproc.py and autopreproc.py

Try running,
```bash
python input.py --nodes=1 --directory=.
```

### setup.py
Controlls which detectors and what pre-analysis is performed during the preprocessing.

## Batch analysis info from LCLS
Info on the available batch system and how to submit batch jobs can be found on the [LCLS confluence page](https://confluence.slac.stanford.edu/display/PCDS/Batch+System+Analysis+Jobs?src=sidebar).

