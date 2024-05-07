import multiprocessing

def acquire_semaphore(semaphore):
    try:
        semaphore.acquire()
        # Do something with the semaphore
    finally:
        semaphore.release()

if __name__ == '__main__':
    semaphore = multiprocessing.Semaphore()
    acquire_semaphore(semaphore)
