#!/usr/bin/env python
import os,sys
# sys.path.insert(0,os.path.join(os.getcwd(),'.'))
# import pylab, matplotlib
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

def main(*args):

    proposed = [850, 9, 67, 29, 41, 27, 35, 45, 36, 73, 66]
    magic_scissors = [850, 890, 852, 786, 758, 767, 788, 818, 770, 726, 614]

    plt.figure()
    plt.xlim(1,11)
    plt.plot(range(1, len(proposed)+1), proposed, '-ro', label='Proposed')
    plt.plot(range(1, len(magic_scissors)+1), magic_scissors, '-go', label='Magic Scissors')
    plt.legend()
    plt.xlabel('Slice')
    plt.ylabel('Number of Clicks')
    plt.savefig('interactive_eval.pdf')

    def to_minutes(t):
        m,s = divmod(t,60)
        return float(m)+float(s/60.0)

    proposed_time = map(to_minutes,[1655, 76, 215, 124, 186, 110, 155, 293, 141, 284, 259])
    magic_scissors_time = map(to_minutes,[1655, 1562, 1300, 1069, 1039, 1016, 1021, 1042, 1079, 950, 834])

    plt.figure()
    plt.xlim(1,11)
    plt.plot(range(1, len(proposed_time)+1), proposed_time, '-ro', label='Proposed')
    plt.plot(range(1, len(magic_scissors_time)+1), magic_scissors_time, '-go', label='Magic Scissors')
    plt.legend()
    plt.xlabel('Slice')
    plt.ylabel('Minutes')
    plt.savefig('interactive_eval_time.pdf')

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
