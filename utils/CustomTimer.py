# Custom timer
import time

class CheckTime():
    def __init__(self):
        self.start_time = time.time()
    
    def get_start_time_gm(self):
        return time.gmtime(self.start_time)
    
    def get_start_time_str(self):
        return time.strftime("%H:%M:%S", self.start_time)

    def get_running_time_gm(self):
        return time.gmtime(ime.time() - self.start_time)
    
    def get_running_time_flt(self):
        return time.time() - self.start_time
    
    
    def get_running_time_str(self):
        running_time = time.gmtime(time.time() - self.start_time)
        return time.strftime("%H:%M:%S",running_time)
        
    
