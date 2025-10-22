#from LSW import Atom


import time
import tracemalloc

USE_MALLOC = False  # Set to False to disable memory tracking

class Debug_timing:
    def __init__(self):
        self.ref_timestamp = int(time.time() * 1000)
        self.start_timestamp = int(time.time() * 1000)
        self.debug_info = ""
        if USE_MALLOC:
            tracemalloc.start()  # start tracking memory
            self.last_snapshot = tracemalloc.take_snapshot()
        else:
            self.last_snapshot = None
    def print_complete_time_since_start(self):
        new_timestamp = int(time.time() * 1000)
        time_diff = new_timestamp - self.start_timestamp
        
        with open("debug_timing_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"Total time: {time_diff}ms \n")
    def print_difference(self, debug_info, do_print=True):
        new_timestamp = int(time.time() * 1000)
        time_diff = new_timestamp - self.ref_timestamp
        self.ref_timestamp = new_timestamp

        mem_diff_kb = 0.0
        if USE_MALLOC and self.last_snapshot:
            # Take new snapshot and compare
            current_snapshot = tracemalloc.take_snapshot()
            stats = current_snapshot.compare_to(self.last_snapshot, 'lineno')
            self.last_snapshot = current_snapshot

            # Total memory difference (bytes)
            mem_diff = sum([stat.size_diff for stat in stats])
            mem_diff_kb = mem_diff / 1024

        if do_print:
            print(f"{time_diff}ms : {debug_info} | Δ RAM: {mem_diff_kb:+.1f} KB")
            with open("debug_timing_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(f"{time_diff}ms | Δ RAM: {mem_diff_kb:+.1f} KB --> {debug_info} \n")

        return new_timestamp

    def tolist(self):
        return []

if __name__ == "__main__":
    import LSW
    LSW.call_main()