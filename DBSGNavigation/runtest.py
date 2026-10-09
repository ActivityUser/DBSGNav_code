from runners.test_runner import test_runner
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def main():
    runner = test_runner()
    runner.train()

if __name__ == '__main__':
    main()