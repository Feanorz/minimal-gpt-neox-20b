import torch.nn as nn
import torch
import time
from torch.multiprocessing import Process, Queue


# Maps map_fn, onto args, n times
def q_map(map_fn, n, args):
    procs = []
    for i in range(n):
        # Make sure main process has enough time to load back output before destroying this function
        mp_queue, sync_queue = Queue(), Queue()

        proc = Process(target=execute_fun, args=(mp_queue, sync_queue, map_fn, args))
        proc.start()

        procs.append((proc, mp_queue, sync_queue))

    results = []
    for proc, mp_queue, sync_queue in procs:
        results.append(mp_queue.get())

        sync_queue.put(1)

    return results


def execute_fun(mp_queue, sync_queue, map_fn, args):
    output = map_fn(*args)
    mp_queue.put(output)
    # Close function
    sync_queue.get()



class myClass:
    def __init__(self, a, b):
        print(a + b)



x = q_map(myClass, 5, (5, 6))
print(x)
