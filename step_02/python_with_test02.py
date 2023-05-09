# 使用contextlib测试with
import contextlib


@contextlib.contextmanager
def config_test():
    print("start")
    try:
        yield
    finally:
        print("done")


with config_test():
    print("process")
