import sys


def flat2d(lst):
    return sum(lst, [])


class SkipWithBlock(Exception):
    pass


class Section:
    def __init__(self, *args, skip=False):
        self.skip = skip

    def __enter__(self):
        if self.skip:
            sys.settrace(lambda *args, **keys: None)
            frame = sys._getframe(1)
            frame.f_trace = self.trace
    
    def trace(self, frame, event, arg):
        raise SkipWithBlock()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            return  # No exception
        if issubclass(exc_type, SkipWithBlock):
            return True  # Suppress special SkipWithBlock exception

    def step(self):
        pass
    
    
def Todo(*args):
    pass


def split_array_into_chunks(arr, n):
    if n <= 0:
        return "Number of chunks should be greater than 0"

    chunk_size = len(arr) // n
    chunks = [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]

    # Adjusting last chunk in case of uneven division
    if len(chunks) > n:
        chunks[-2] += chunks[-1]
        chunks = chunks[:-1]

    return chunks


def to_item(v):
    if hasattr(v, 'item'):
        return v.item()
    else:
        return v
    
def detach(v):
    if hasattr(v, 'detach'):
        return v.detach()
    else:
        return v
    
    
def color_terms(s, words):
    for w in words:
        s = s.replace(w, f"\033[41m\033[97m{w}\033[0m")
    return s



def simple_mean(v):
    if len(v) == 0: return 0
    else: return sum(v) / len(v)