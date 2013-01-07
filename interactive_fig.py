#!/usr/bin/env python
import os,sys
sys.path.insert(0,os.path.join(os.getcwd(),'.'))
# import pylab, matplotlib
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

import figure as f

def main(*args):

    proposed = [850, 9, 67, 29, 41, 27, 35, 45, 36, 73, 66]
    magic_scissors = [850, 890, 852, 786, 758, 767, 788, 818, 770, 726, 614]
    hybrid = [850, 183, 188, 153, 264, 231, 247, 258, 276, 308, 241]
    auto = [850, 6, 9, 10, 12, 8, 9, 16, 16, 18, 24]

    plt.figure()
#    plt.xlim(0,11)
    plt.ylim(0,1000)
    # plt.yscale('log')
    plt.plot(range(0, len(proposed)), proposed, '-ro', label='Proposed')
    plt.plot(range(0, len(auto)), auto, '-mo', label='Proposed + Parameter Estimation')
    plt.plot(range(0, len(magic_scissors)), magic_scissors, '-go', label='Intelligent Scissors')
    plt.plot(range(0, len(hybrid)), hybrid, '-bo', label='Intelligent Scissors + Propagation')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13))
    plt.xlabel('Slice')
    plt.ylabel('Number of Clicks')
    plt.savefig('interactive_eval.pdf')

    def to_minutes(t):
        m,s = divmod(t,60)
        return float(m)+float(s/60.0)

    proposed_time = map(to_minutes,[1655, 76, 215, 124, 186, 110, 155, 293, 141, 284, 259])
    magic_scissors_time = map(to_minutes,[1655, 1562, 1300, 1069, 1039, 1016, 1021, 1042, 1079, 950, 834])
    hybrid_time = map(to_minutes,[1655, 252, 207, 180, 229, 266, 235, 239, 252, 298, 265])
    auto_time = map(to_minutes, [1655, 44, 57, 50, 61, 44, 56, 71, 67, 72, 90])

    plt.figure()
#    plt.xlim(1,11)
    plt.plot(range(0, len(proposed_time)), proposed_time, '-ro', label='Proposed')
    plt.plot(range(0, len(auto_time)), auto_time, '-mo', label='Proposed + Parameter Estimation')
    plt.plot(range(0, len(magic_scissors_time)), magic_scissors_time, '-go', label='Intelligent Scissors')
    plt.plot(range(0, len(hybrid_time)), hybrid_time, '-bo', label='Intelligent Scissors + Propagation')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13))
    plt.xlabel('Slice')
    plt.ylabel('Time (Minutes)')
    plt.savefig('interactive_eval_time.pdf')

    iscores = map(f.read_score, [ 'seq1/interactive/image{0:04d}.score'.format(i) for i in range(91,101) ])
    pscores = map(f.read_score, [ 'seq1/global-20/90/image{0:04d}.score'.format(i) for i in range(91,101) ])

    ilabel = 'Proposed Interactive Segmention'
    plabel = 'Previous Automatic Method'

    def precision_fig(field,label):
        plt.figure()
#        plt.xlim(1,10)
        plt.plot(range(1, 11), [ i._asdict()[field] for i in iscores ], '-ro', label=ilabel)
        plt.plot(range(1, 11), [ i._asdict()[field] for i in pscores ], '-bo', label=plabel)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13))
        plt.xlabel('Slice')
        plt.ylabel(label)
        plt.savefig('interactive_'+field+'.pdf')

    precision_fig('f','F-measure')
    precision_fig('p','Precision')
    precision_fig('r','Recall')

    # plt.figure()
    # plt.xlim(1,10)
    # iscores = map(f.read_score, [ 'seq1/interactive/image{0:04d}.score'.format(i) for i in range(91,101) ])
    # pscores = map(f.read_score, [ 'seq1/global-20/90/image{0:04d}.score'.format(i) for i in range(91,101) ])
    # plt.plot(range(1, 11), [ i.p for i in iscores ], '-ro', label=ilabel)
    # plt.plot(range(1, 11), [ i.p for i in pscores ], '-bo', label=plabel)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13))
    # plt.xlabel('Slice')
    # plt.ylabel('Precision')
    # plt.savefig('interactive_p.pdf')

    # plt.figure()
    # plt.xlim(1,10)
    # iscores = map(f.read_score, [ 'seq1/interactive/image{0:04d}.score'.format(i) for i in range(91,101) ])
    # pscores = map(f.read_score, [ 'seq1/global-20/90/image{0:04d}.score'.format(i) for i in range(91,101) ])
    # plt.plot(range(1, 11), [ i.r for i in iscores ], '-ro', label=ilabel)
    # plt.plot(range(1, 11), [ i.r for i in pscores ], '-bo', label=plabel)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13))
    # plt.xlabel('Slice')
    # plt.ylabel('Recall')
    # plt.savefig('interactive_r.pdf')

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
