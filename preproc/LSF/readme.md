# TMO preprocessing
Contributors: Elio Champenois, Taran Driver, Matt Ware

## Required libraries
You'll need to source the newest psana2 libraries. Run `source /cds/sw/ds/ana/conda2/manage/bin/psconda.sh` on the terminal.

## Mandatory updates
* Batch submission is currently using the old LSF (bsub) interface. This has now been taken down and will no longer work.

## Suggested updates
* Experiment is hard coded into `preproc.py`. This should be handed to the script from the command line.
* Output folder should also be a command line prompt.

## Hard coded values to update
First off, these really should be removed and better implemented, but as it stands ... Here is what needs to be updated before running the script.

`preproc.py`
* `exp` on line 13
* `preprocessed_folder` on line 16
* ion tof settings on lines 29 to 35
* detectors inside of the loop may need to be updated
* detector logic may need to be updated

`bsub.sh`
* `ver` on line 4
* `base_path` on line 5
* `script` on line 6
* `log` on line 7
* `server` on line 8

`autopreproc.py`
* `run`
* `run_stop`
* `sh_folder`
* `sh_file`


## Preprocessing flow
1) Automated job submission

	* Controlled by `autopreproc.py`, which calls `bsub.py`.

	* Update the following lines before submission:
		```python
		run = 185
		run_stop = 191
		
		sh_folder = "/reg/d/psdm/tmo/tmolv2918/results/elio/preproc/"
		sh_file = "bsub_v0.sh"
		```
	* Run via `python autopreproc.py`

2) Single job submission
	* Controlled by `preproc.py`

## File descriptions

### bsub.sh
Control the batch submission, calls preproc.py

### autopreproc.py
Automates the batch submission, calls bsub.sh

### slurmJob.sh
Submits SLURM jobs. Try running first with psdebugq.

### preproc.py

### input.py
Specifies the command line input structure that is read in by preproc.py

Try running,
```bash
python input.py --nodes=1 --directory=.
```

## Batch analysis info from LCLS
Info on the available batch system and how to submit batch jobs can be found on the [LCLS confluence page](https://confluence.slac.stanford.edu/display/PCDS/Batch+System+Analysis+Jobs?src=sidebar).

## SLURM
