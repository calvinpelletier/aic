from time import time


def measure_data_speed(it, n_batches=1000):
    start = time()
    for i, batch in enumerate(it):
        if i >= n_batches:
            break
    end = time()
    return n_batches / (end - start)
