import time

class TickTimer:
    def __init__(self, start_key=None):
        self.T = time.time()
        self.time_map = {}
        self.current_key = start_key

    def tick(self, new_key):
        lapse = time.time() - self.T
        if self.current_key is not None:
            if self.current_key not in self.time_map:
                self.time_map[self.current_key] = 0
            self.time_map[self.current_key] += lapse
        
        self.current_key = new_key
        self.T = time.time()

    def reset(self):
        self.time_map = {}
        self.T = time.time()

    def report(self):
        if self.current_key is not None:
            self.tick(None)
        print("[TIMING REPORT] %s" % (" ".join("[%s: %.5f sec]" % (k, v) for k, v in self.time_map.items())))
