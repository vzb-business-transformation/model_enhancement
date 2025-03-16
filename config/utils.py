import time

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elasped_time = end_time - start_time
        if elasped_time < 60:
            print(f'{func.__name__} took {elasped_time:.2f} seconds to run.')
        else:
            elasped_time_mins = elasped_time/60
            print(f'{func.__name__} took {elasped_time_mins:.2f} minutes to run.')
        return result
    return wrapper
