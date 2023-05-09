def get_sample():
    return Sample()


class Sample:
    def __enter__(self):
        print("__enter__")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("__exit__")
        print(exc_type)
        print(exc_val)
        print(exc_tb)

    def do_something(self):
        a = 1 / 0
        return a


def config_test():
    print("start")
    try:
        yield
    finally:
        print("done")


if __name__ == '__main__':
    # with get_sample() as sample:
    #     sample.do_something()

    with config_test():
        print("process")


# 使用contextlib测试with





