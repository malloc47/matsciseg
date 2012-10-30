#!/usr/bin/env python
import sys,os
sys.path.insert(0,os.path.join(os.getcwd(),'..'))

def main(*args):
    if(len(args) < 2):
        return 1

    prefix = args[1]

    exe       = '/home/malloc47/src/projects/matsci/matsciskel/matsciskel.py'
    ws_exe    = '/home/malloc47/src/projects/matsci/matsciskel/watershed.py'
    data_path = '/home/malloc47/src/projects/matsci/matsciskel'

    datasets = []
    name_to_dir = lambda n: 'syn/'+n

    dilation = 20

    def global_cmd(n,rn,i,j,r):
        if rn==i:
            return """ mkdir -p {7}/{0}/{4}/{1:d}/
    ln -s ../../ground/{2:04d}.label {7}/{0}/{4}/{1:d}/{2:04d}.label
    {6} e global {7}/{0}/img/{2:04d}.png {7}/{0}/ground/{2:04d}.label {7}/{0}/img/{3:04d}.png {7}/{0}/{4}/{1:d}/{3:04d}.label {7}/{0}/{4}/{1:d}/{3:04d}.png {5}
    """.format(n,rn,i,j,r,dilation,exe,data_path)
        else:
            return """{6} e global {7}/{0}/img/{2:04d}.png {7}/{0}/{4}/{1:d}/{2:04d}.label {7}/{0}/img/{3:04d}.png {7}/{0}/{4}/{1:d}/{3:04d}.label {7}/{0}/{4}/{1:d}/{3:04d}.png {5}
    """.format(n,rn,i,j,r,dilation,exe,data_path)

    def global_local_cmd(n,rn,i,j,r):
        if rn==i:
            return """mkdir -p {7}/{0}/{4}/{1:d}/
    ln -s ../../ground/{2:04d}.label {7}/{0}/{4}/{1:d}/{2:04d}.label
    {6} e global {7}/{0}/img/{2:04d}.png {7}/{0}/ground/{2:04d}.label {7}/{0}/img/{3:04d}.png {7}/{0}/{4}/{1:d}/{3:04d}-old.label {7}/{0}/{4}/{1:d}/{3:04d}.png {5}
    LD_LIBRARY_PATH=/home/malloc47/src/programs/OpenCV-2.0.0/build/lib /home/malloc47/src/projects/matsci/matscicut-debian/matscicut {7}/{0}/img/{3:04d}.png {7}/{0}/{4}/{1:d}/{3:04d}-old.label {7}/{0}/{4}/{1:d}/{3:04d}.label
    """.format(n,rn,i,j,r,dilation,exe,data_path)
        else:
            return """{6} e global {7}/{0}/img/{2:04d}.png {7}/{0}/{4}/{1:d}/{2:04d}.label {7}/{0}/img/{3:04d}.png {7}/{0}/{4}/{1:d}/{3:04d}-old.label {7}/{0}/{4}/{1:d}/{3:04d}.png {5}
    LD_LIBRARY_PATH=/home/malloc47/src/programs/OpenCV-2.0.0/build/lib /home/malloc47/src/projects/matsci/matscicut-debian/matscicut {7}/{0}/img/{3:04d}.png {7}/{0}/{4}/{1:d}/{3:04d}-old.label {7}/{0}/{4}/{1:d}/{3:04d}.label
    """.format(n,rn,i,j,r,dilation,exe,data_path)

    def local_cmd(n,rn,i,j,r):
        if rn==i:
            return """mkdir -p {7}/{0}/{4}/{1:d}/
    ln -s ../../ground/{2:04d}.label {7}/{0}/{4}/{1:d}/{2:04d}.label
    {6} e clique {7}/{0}/img/{2:04d}.png {7}/{0}/ground/{2:04d}.label {7}/{0}/img/{3:04d}.png {7}/{0}/{4}/{1:d}/{3:04d}.label {7}/{0}/{4}/{1:d}/{3:04d}.png {5}
    """.format(n,rn,i,j,r,dilation,exe,data_path)
        else:
            return """{6} e clique {7}/{0}/img/{2:04d}.png {7}/{0}/{4}/{1:d}/{2:04d}.label {7}/{0}/img/{3:04d}.png {7}/{0}/{4}/{1:d}/{3:04d}.label {7}/{0}/{4}/{1:d}/{3:04d}.png {5}
    """.format(n,rn,i,j,r,dilation,exe,data_path)

    def local_local_cmd(n,rn,i,j,r):
        if rn==i:
            return """mkdir -p {7}/{0}/{4}/{1:d}/
    ln -s ../../ground/{2:04d}.label {7}/{0}/{4}/{1:d}/{2:04d}.label
    {6} e clique {7}/{0}/img/{2:04d}.png {7}/{0}/ground/{2:04d}.label {7}/{0}/img/{3:04d}.png {7}/{0}/{4}/{1:d}/{3:04d}-old.label {7}/{0}/{4}/{1:d}/{3:04d}.png {5}
    LD_LIBRARY_PATH=/home/malloc47/src/programs/OpenCV-2.0.0/build/lib /home/malloc47/src/projects/matsci/matscicut-debian/matscicut {7}/{0}/img/{3:04d}.png {7}/{0}/{4}/{1:d}/{3:04d}-old.label {7}/{0}/{4}/{1:d}/{3:04d}.label
    """.format(n,rn,i,j,r,dilation,exe,data_path)
        else:
            return """{6} e clique {7}/{0}/img/{2:04d}.png {7}/{0}/{4}/{1:d}/{2:04d}.label {7}/{0}/img/{3:04d}.png {7}/{0}/{4}/{1:d}/{3:04d}-old.label {7}/{0}/{4}/{1:d}/{3:04d}.png {5}
    LD_LIBRARY_PATH=/home/malloc47/src/programs/OpenCV-2.0.0/build/lib /home/malloc47/src/projects/matsci/matscicut-debian/matscicut {7}/{0}/img/{3:04d}.png {7}/{0}/{4}/{1:d}/{3:04d}-old.label {7}/{0}/{4}/{1:d}/{3:04d}.label
    """.format(n,rn,i,j,r,dilation,exe,data_path)

    # names = [ 'd1s'+str(i) for i in range(1,13) ]
    # stdlen = range(0,50)
    # ground_slices = [5,15,25,35,45]
    # rng = None
    # datasets += [ (n,'global-'+str(dilation),stdlen,ground_slices,global_cmd,rng) for n in names]
    # datasets += [ (n,'global-local-'+str(dilation),stdlen,ground_slices,global_local_cmd,rng) for n in names]
    # datasets += [ (n,'local-'+str(dilation),stdlen,ground_slices,local_cmd,rng) for n in names]
    # datasets += [ (n,'local-local-'+str(dilation),stdlen,ground_slices,local_local_cmd,rng) for n in names]

    # names = [ 'd1s'+str(i) for i in range(13,15) ]
    # stdlen = range(0,500)
    # ground_slices = range(0,500,50)+[499]
    # rng = None
    # datasets += [ (n,'global-'+str(dilation),stdlen,ground_slices,global_cmd,rng) for n in names]
    # datasets += [ (n,'global-local-'+str(dilation),stdlen,ground_slices,global_local_cmd,rng) for n in names]
    # datasets += [ (n,'local-local-'+str(dilation),stdlen,ground_slices,local_local_cmd,rng) for n in names]
    # datasets += [ (n,'local-'+str(dilation),stdlen,ground_slices,local_cmd,rng) for n in names]

    names = [ 'd1s'+str(i) for i in range(16,20) ]
    stdlen = range(0,300)
    ground_slices = [i for i in range(0,300,50)+[299] if i != 150]
    rng = 50
    datasets += [ (n,'global-'+str(dilation),stdlen,ground_slices,global_cmd,rng) for n in names]
    datasets += [ (n,'global-local-'+str(dilation),stdlen,ground_slices,global_local_cmd,rng) for n in names]
    datasets += [ (n,'local-local-'+str(dilation),stdlen,ground_slices,local_local_cmd,rng) for n in names]
    datasets += [ (n,'local-'+str(dilation),stdlen,ground_slices,local_cmd,rng) for n in names]

    rng = None
    ground_slices = [150]
    datasets += [ (n,'global-'+str(dilation),stdlen,ground_slices,global_cmd,rng) for n in names]
    datasets += [ (n,'global-local-'+str(dilation),stdlen,ground_slices,global_local_cmd,rng) for n in names]
    datasets += [ (n,'local-local-'+str(dilation),stdlen,ground_slices,local_local_cmd,rng) for n in names]
    datasets += [ (n,'local-'+str(dilation),stdlen,ground_slices,local_cmd,rng) for n in names]

    dilation = 10

    def watershed_cmd(n,rn,i,j,r):
        if rn==i:
            return """mkdir -p {7}/{0}/{4}/{1:d}/
ln -s ../../ground/{2:04d}.label {7}/{0}/{4}/{1:d}/{2:04d}.label
{6} {7}/{0}/img/{3:04d}.png {7}/{0}/ground/{2:04d}.label {7}/{0}/{4}/{1:d}/{3:04d}.label 3 {5}
    """.format(n,rn,i,j,r,dilation,ws_exe,data_path)
        else:
            return """{6} {7}/{0}/img/{3:04d}.png {7}/{0}/{4}/{1:d}/{2:04d}.label {7}/{0}/{4}/{1:d}/{3:04d}.label 3 {5}
    """.format(n,rn,i,j,r,dilation,ws_exe,data_path)

    names = [ 'd1s'+str(i) for i in range(1,13) ]
    stdlen = range(0,50)
    ground_slices = [5,15,25,35,45]
    rng = None
    datasets += [ (n,'watershed-'+str(dilation),stdlen,ground_slices,watershed_cmd,rng) for n in names]

    names = [ 'd1s'+str(i) for i in range(16,20) ]
    stdlen = range(0,300)
    ground_slices = [150]
    rng = None
    datasets += [ (n,'watershed-'+str(dilation),stdlen,ground_slices,watershed_cmd,rng) for n in names]

    for n,run,slices,gt,cmd,rng in datasets:
        for g in gt:
            if not rng is None:
                forward =  [ e for e in slices if e>=g and e-g < rng+1]
                backward = list(reversed([ e for e in slices if e<=g and g-e < rng+1]))
            else:
                forward =  [ e for e in slices if e>=g]
                backward = list(reversed([ e for e in slices if e<=g ]))
            forward_file = prefix+n+'-'+run+'-'+str(g)+'-f.sh'
            backward_file = prefix+n+'-'+run+'-'+str(g)+'-b.sh'
            with open(forward_file, 'wb') as f:
                f.write('#!/bin/bash\n')
                for i, j in zip(forward[:-1], forward[1:]):
                    f.write(cmd(name_to_dir(n),g,i,j,run))
            with open(backward_file, 'wb') as f:
                f.write('#!/bin/bash\n')
                for i, j in zip(backward[:-1], backward[1:]):
                    f.write(cmd(name_to_dir(n),g,i,j,run))

# ./matsciskel.py e global seq1/img/image0090.png seq1/ground/image0090.label seq1/img/image0091.png test2.label test2.png 10

if __name__ == '__main__':
    sys.exit(main(*sys.argv))
