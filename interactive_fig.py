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

    syn_clicks = { 'magic' : [210, 148, 146, 123, 182, 170, 164, 122, 131, 137, 149, 143, 160, 135, 171, 145, 131, 120, 129, 127],
                   'magicprop' : [0,13, 14, 21, 36, 20, 26, 28, 38, 47, 40, 45, 50, 38, 46, 40, 52, 31, 31, 32],
                   'proposed' : [0, 1, 1, 4, 7, 5, 6, 5, 4, 8, 8, 13, 8, 14, 21, 13, 9, 21, 19, 12],
                   'propagation' : [0,0,0,0,0,0,0,0,0,0,13,5,11,0,0,0,0,0,0,0],

                   }

    syn_time = { 'magic' : map(to_minutes,[415, 306, 307, 247, 289, 403, 337, 188, 227, 267, 245, 225, 244, 202, 214, 225, 205, 172, 182, 196]),
                 'magicprop' : map(to_minutes,[0,45, 37, 36, 64, 38, 45, 46, 49, 71, 46, 68, 104, 70, 72, 67, 88, 48, 57, 49]),

                 'proposed' : map(to_minutes,[0, 7, 7, 16, 14, 14, 20, 16, 17, 29, 20, 40, 23, 40, 61, 42, 32, 70, 66, 48]),
                 'propagation' : map(to_minutes,[0,0,0,0,0,0,0,0,0,0,42,29,33,0,0,0,0,0,0,0])}

    plt.figure()
    plt.xlim(0,19)
    plt.ylim(0,220)
    # plt.yscale('log')
    # plt.plot(range(0, len(proposed)), proposed, '-ro', label='Proposed')
    plt.plot(range(0, len(syn_clicks['proposed'])), syn_clicks['proposed'], '-ro', label='Proposed + Parameter Estimation')
    plt.plot(range(0, len(syn_clicks['propagation'])), syn_clicks['propagation'], '-mo', label='Proposed + Repropagation')
    plt.plot(range(0, len(syn_clicks['magic'])), syn_clicks['magic'], '-go', label='Intelligent Scissors')
    plt.plot(range(0, len(syn_clicks['magicprop'])), syn_clicks['magicprop'], '-bo', label='Intelligent Scissors + Propagation')

    # plt.plot(range(0, len(hybrid)), hybrid, '-bo', label='Intelligent Scissors + Propagation')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13))
    plt.xlabel('Synthetic Slice Number')
    plt.ylabel('Number of Clicks')
    plt.savefig('interactive_eval_syn.pdf')

    plt.figure()
    plt.xlim(0,19)
    plt.ylim(0,8)

    plt.plot(range(0, len(syn_time['proposed'])), syn_time['proposed'], '-ro', label='Proposed + Parameter Estimation')
    plt.plot(range(0, len(syn_time['propagation'])), syn_time['propagation'], '-mo', label='Proposed + Repropagation')
    plt.plot(range(0, len(syn_time['magic'])), syn_time['magic'], '-go', label='Intelligent Scissors')
    plt.plot(range(0, len(syn_time['magicprop'])), syn_time['magicprop'], '-bo', label='Intelligent Scissors + Propagation')

    # plt.plot(range(0, len(proposed_time)), proposed_time, '-ro', label='Proposed')
    # plt.plot(range(0, len(auto_time)), auto_time, '-mo', label='Proposed + Parameter Estimation')
    # plt.plot(range(0, len(magic_scissors_time)), magic_scissors_time, '-go', label='Intelligent Scissors')
    # plt.plot(range(0, len(hybrid_time)), hybrid_time, '-bo', label='Intelligent Scissors + Propagation')

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13))
    plt.xlabel('Slice')
    plt.ylabel('Time (Minutes)')
    plt.savefig('interactive_eval_time_syn.pdf')

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
