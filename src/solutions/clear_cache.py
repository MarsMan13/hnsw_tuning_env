from joblib import Memory

if __name__ == "__main__":
    print("Clearing caches...")
    for cache_dir in [
            "/tmp/brute_force_cache",
            "/tmp/random_search_cache",
            "/tmp/grid_search_cache",
            "/tmp/our_solution_cache",
            "/tmp/vd_tuner_cache",
        ]:
        memory = Memory(cache_dir, verbose=0)
        print(f"Clearing cache at {cache_dir}...")
        memory.clear(warn=False)
        print(f"Cache cleared at {cache_dir}!\n")