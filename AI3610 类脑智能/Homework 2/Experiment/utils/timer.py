import time


def get_time_stamp():
    return time.strftime('%Y%m%d%H%M%S', time.localtime())


def get_time_text():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
