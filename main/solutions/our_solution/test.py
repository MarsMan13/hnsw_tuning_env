def cpu_bound_task(x):
    print(f"PID: {os.getpid()} - Start {x}")
    total = 0
    for i in range(10**20):
        total += i ** 0.5
    print(f"PID: {os.getpid()} - Done {x}")
    return total

if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    import os

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(cpu_bound_task, i) for i in range(8)]
        for f in futures:
            f.result()
