# xtc provides scripts and methods for reading the xtc files

## Jupyter Notebooks
### read_raw_streamlined.ipynb
Similar to read_raw, but it provides a convenient obfuscation of the loop structure to access the data.
Other notebooks below, provide similar functionality in a more clunky approach.

### read_raw.ipynb
Reads in the xtc files serially.
Provides examples of how to setup detectors and determine how to access their data content.
Shows some example analyses.

### H5Writer_debug.ipynb
Debugging version for writing H5 files.
Useful to see what `data.py` does behind the scenes.

### H5Writer_serial.ipynb
Serial version of H5 writer.
Runs at about 60 Hz.
Useful for trialling new analyses.

### UV_diode_tester.ipynb
Provides an example to test that the goose trigger is working.

## Libraries
### analyses.py
Class for generated run data from the individual shot data.
User specificied analysis is performed on the shot data, then the data is saved shot-by-shot.

### data.py
Function for reading from an XTC file.
Requres a detector and analysis dictionary from the user as demonstrated in `read_raw_streamlined.ipynb`.

### loop.py
Convenience library for adding modifications to iterators in python.
For example, the code snipped below add execution time to a simple loop.
```python
import time
def timeIt(iterator, printEverySec = 10):
    tic = time.time()
    totalTime = 0
    for idx, el in enumerate(iterator):
        toc = time.time()
        elapsedTimeSec = toc-tic
        if elapsedTimeSec > printEverySec:
            totalTime += elapsedTimeSec
            tic=toc
            print('%f sec passed, %d iterations' % (totalTime, idx))
        yield el
        
for idx in timeIt( range(100), printEverySec=10):
  time.sleep(1)
```