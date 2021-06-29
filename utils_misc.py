import numpy as np, os, time, tqdm
from datetime import datetime
from colorama import Fore
import sys

# GPU-related business
def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_smi')
    memory_available = [int(x.split()[2])+5*i for i, x in enumerate(open('tmp_smi', 'r').readlines())]
    os.remove("tmp_smi")
    return np.argmax(memory_available)

def any_gpu_with_space(gb_needed):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_smi')
    memory_available = [float(x.split()[2])/1024.0 for i, x in enumerate(open('tmp_smi', 'r').readlines())]
    os.remove("tmp_smi")
    return any([mem >= gb_needed for mem in memory_available])

def wait_free_gpu(gb_needed):
    while not any_gpu_with_space(gb_needed):
        time.sleep(30)

def select_freer_gpu():
    freer_gpu = str(get_freer_gpu())
    print("Will use GPU: %s" % (freer_gpu))
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""+freer_gpu
    return freer_gpu

class DoublePrint(object):
    def __init__(self, name, show_timings=False):
        self.file = open(name, "a")
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.show_timings = show_timings
        sys.stderr = self
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.file.close()

    def write(self, data):
        if self.show_timings:
            data = str(data)
            if len(data.strip()) > 0:
                data = Fore.LIGHTBLUE_EX+"["+datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"] " +Fore.RESET+data

        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()

    def flush(self):
        self.file.flush()

def batcher(iterator, batch_size=4, progress=False):
    if progress:
        iterator = tqdm.tqdm(iterator)

    batch = []
    for elem in iterator:
        batch.append(elem)
        if len(batch) == batch_size:
            final_batch = batch
            batch = []
            yield final_batch
    if len(batch) > 0: # Leftovers
        yield batch
