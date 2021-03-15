import argparse

def initArg( args, stringSpec , valType ):
    val = valType( eval(str('args.') + stringSpec) )
    print(('%s read as '%(stringSpec))+str(val))
    return val

parser = argparse.ArgumentParser()
parser.add_argument("--cores", help="Number of cores to parallelize across.", default=1)
parser.add_argument("--run", help="Run to analyze.", default=1)
parser.add_argument("--exp", help="Experiment to analyze.", default='tmolv2918')
parser.add_argument("--queue", help="Queue to use.", default='psfehq')
parser.add_argument("--runStop", help="Last run to analyze.", default=-1)
parser.add_argument("--nevent", help="Events to analyze.", default=100000)
parser.add_argument("--directory", help="Results will be save to <experiment dir>/scratch/<username>/<preproc>/<directory>")
parser.add_argument("--isDebug", help="Debug mode", default=False)
args = parser.parse_args()

cores = initArg( args, 'cores', int )
directory = initArg( args, 'directory', str )
isDebug = initArg( args, 'isDebug', bool )
exp = initArg( args, 'exp', str )
runNumber = initArg( args, 'run', int )
nevents = initArg(args, 'nevent', int)
runStop = initArg(args, 'runStop', int)
queue = initArg(args, 'queue', str)