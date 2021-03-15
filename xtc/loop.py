import time


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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

def timeItNode(iterator, printEverySec = 10):
    tic = time.time()
    totalTime = 0
    for idx, el in enumerate(iterator):
        toc = time.time()
        elapsedTimeSec = toc-tic
        if elapsedTimeSec > printEverySec:
            totalTime += elapsedTimeSec
            tic=toc
            print('%d node, %f sec passed, %d iterations' % (rank, totalTime, idx))
        yield el
        
def moduloSkip(iterator, modulo=2):
    for idx, el in enumerate(iterator):
        if idx%modulo == 0:
            yield el
            
def wrap(iterator):
    return timeIt(iterator, printEverySec=10)