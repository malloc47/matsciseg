#!/usr/bin/env python
import sys,os,stat
sys.path.insert(0,os.path.join(os.getcwd(),'..'))
import functools

def main(*args):
    if(len(args) < 2):
        return 1

    prefix = args[1]

    exe       = '/home/malloc47/src/projects/matsci/matsciskel/matsciskel.py'
    ws_exe    = '/home/malloc47/src/projects/matsci/matsciskel/watershed.py'
    data_path = '/home/malloc47/src/projects/matsci/matsciskel'

    datasets = []

    def seq1_cmd(exe,data_path,name,edge_type,seg_type,dilation,fprefix,run,rn,i,j):
        if rn==i:
            return """mkdir -p {7}/{0}/{4}/{1:d}/
    ln -s ../../ground/{10}{2:04d}.label {7}/{0}/{4}/{1:d}/{10}{2:04d}.label
    {6} {8} {9} {7}/{0}/img/{10}{2:04d}.png {7}/{0}/ground/{10}{2:04d}.label {7}/{0}/img/{10}{3:04d}.png {7}/{0}/{4}/{1:d}/{10}{3:04d}-old.label {7}/{0}/{4}/{1:d}/{10}{3:04d}.png {5}
    LD_LIBRARY_PATH=/home/malloc47/src/programs/OpenCV-2.0.0/build/lib /home/malloc47/src/projects/matsci/matscicut-debian/matscicut {7}/{0}/img/{10}{3:04d}.png {7}/{0}/{4}/{1:d}/{10}{3:04d}-old.label {7}/{0}/{4}/{1:d}/{10}{3:04d}.label
    """.format(name,rn,i,j,run,dilation,exe,data_path,edge_type,seg_type,fprefix)
        else:
            return """{6} {8} {9} {7}/{0}/img/{10}{2:04d}.png {7}/{0}/{4}/{1:d}/{10}{2:04d}.label {7}/{0}/img/{10}{3:04d}.png {7}/{0}/{4}/{1:d}/{10}{3:04d}-old.label {7}/{0}/{4}/{1:d}/{10}{3:04d}.png {5}
    LD_LIBRARY_PATH=/home/malloc47/src/programs/OpenCV-2.0.0/build/lib /home/malloc47/src/projects/matsci/matscicut-debian/matscicut {7}/{0}/img/{10}{3:04d}.png {7}/{0}/{4}/{1:d}/{10}{3:04d}-old.label {7}/{0}/{4}/{1:d}/{10}{3:04d}.label
    """.format(name,rn,i,j,run,dilation,exe,data_path,edge_type,seg_type,fprefix)

    def seq12_cmd(exe,data_path,name,edge_type,seg_type,d1,d2,d3,fprefix,run,rn,i,j):
        if rn==i:
            return """mkdir -p {7}/{0}/{4}/{1:d}/
    ln -s ../../ground/{10}{2:04d}.label {7}/{0}/{4}/{1:d}/{10}{2:04d}.label
    {6} {8} {9} {7}/{0}/img/{10}{2:04d}.png {7}/{0}/ground/{10}{2:04d}.label {7}/{0}/img/{10}{3:04d}.png {7}/{0}/{4}/{1:d}/{10}{3:04d}.label {7}/{0}/{4}/{1:d}/{10}{3:04d}.png {5} {11} {12}
    """.format(name,rn,i,j,run,d1,exe,data_path,edge_type,seg_type,fprefix,d2,d3)
        else:
            return """{6} {8} {9} {7}/{0}/img/{10}{2:04d}.png {7}/{0}/{4}/{1:d}/{10}{2:04d}.label {7}/{0}/img/{10}{3:04d}.png {7}/{0}/{4}/{1:d}/{10}{3:04d}.label {7}/{0}/{4}/{1:d}/{10}{3:04d}.png {5} {11} {12}
    """.format(name,rn,i,j,run,d1,exe,data_path,edge_type,seg_type,fprefix,d2,d3)

    seq1_global = functools.partial(seq1_cmd,exe,data_path,'seq1','t','global',20,'image','cs-20')
    seq12_global = functools.partial(seq12_cmd,exe,data_path,'seq12','t','log',30,2,5,'image','log-30')

    datasets += [ (seq1_global, range(90,101), range(90,101), None, 'seq1') ]
    datasets += [ (seq12_global, range(769,781), range(769,781), None, 'seq12') ]

    for cmd, slices, gt, rng, name in datasets:

        with open(prefix+name+'-jobs', 'wb') as f:
            f.write('#!/bin/bash\n')
            for g in gt:
                for postfix in ['f','b']:
                    file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             prefix+name+'-'+str(g)+'-'+postfix+'.sh')
                    f.write(file_name+'\n')

        for g in gt:
            if not rng is None:
                forward =  [ e for e in slices if e>=g and e-g < rng+1]
                backward = list(reversed([ e for e in slices if e<=g and g-e < rng+1]))
            else:
                forward =  [ e for e in slices if e>=g]
                backward = list(reversed([ e for e in slices if e<=g ]))
            forward_file = prefix+name+'-'+str(g)+'-f.sh'
            backward_file = prefix+name+'-'+str(g)+'-b.sh'
            with open(forward_file, 'wb') as f:
                f.write('#!/bin/bash\n')
                for i, j in zip(forward[:-1], forward[1:]):
                    f.write(cmd(g,i,j))
            os.chmod(forward_file,os.stat(forward_file).st_mode | stat.S_IEXEC)
            with open(backward_file, 'wb') as f:
                f.write('#!/bin/bash\n')
                for i, j in zip(backward[:-1], backward[1:]):
                    f.write(cmd(g,i,j))
            os.chmod(backward_file,os.stat(backward_file).st_mode | stat.S_IEXEC)

if __name__ == '__main__':
    sys.exit(main(*sys.argv))