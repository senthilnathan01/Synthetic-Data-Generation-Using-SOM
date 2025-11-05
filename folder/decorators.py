import logging
from functools import wraps
from time import time


def timeit(log_level=logging.INFO, alternative_title=None):
    """
      A decorator for measuring the execution time of a function.

      Args:
        log_level (int, optional): The logging level for the time measurement output.
        alternative_title (str, optional): An alternative title to display in the log instead of the function name.

       Returns:
        function: The decorated function.

    """
    def wrap(f):
        @wraps(f)  # keeps the f.__name__ outside the wrapper
        def wrapped_f(*args, **kwargs):
            t0 = time()
            result = f(*args, **kwargs)
            ts = round(time() - t0, 3)

            title = alternative_title or f.__name__
            logging.getLogger().log(
                log_level, " %s took: %f seconds" % (title, ts))
            return result
        return wrapped_f
    return wrap