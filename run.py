import argparse
from subprocess import call
from multiprocessing import Process


def run(arch):
    call(['python', f, arch])


def get_archs():
    l = []
    for i in [1, 2, 4, 10]:
        for j in [4, 16, 64]:
            s = str(j)
            for k in range(1, i):
                s += '-' + str(j)
            l.append(s)
    return l


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str)
    args = parser.parse_args()

    f = 'examples/trpo_gym_{}.py'.format(args.exp)

    archs = get_archs()
    print(archs)

    processes = [Process(target=run, args=(arch,)) for arch in archs]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
