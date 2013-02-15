import matsci
import numpy as np

def global_cmd(arg,im,im_gray,im_prev,seed):
    v = matsci.gco.Slice(im_gray,seed)
    print("Initialized")
    v.data.dilate_all(arg['d'])
    # v.data.output_data_term()
    print("Dilated")
    # v.set_adj_label_all(0);
    # print("Adjacent")
    v.graph_cut(arg['gctype'])
    print("Graph Cut Complete")
    # import code; code.interact(local=locals())
    return v.labels.v

def matrix_cmd(arg,im,im_gray,im_prev,seed):
    v = matsci.gco.Slice(im_gray,seed,bg=True,lightweight=True)
    print("Initialized")
    v.data.dilate_fixed_center(arg['d'],rel_size=0.2,min_size=5,first=False)
    v.data.dilate_first(arg['d'])
    print("Dilated")
    v.adj.set_adj_bg()
    v.graph_cut(arg['gctype'],lite=True)
    prinnnnnt("Graph Cut Complete")
    return v.labels.v

def matrix_unfixed_cmd(arg,im,im_gray,im_prev,seed):
    v = matsci.gco.Slice(im_gray,seed,bg=True,lightweight=True)
    print("Initialized")
    v.data.dilate_all(arg['d'])
    print("Dilated")
    v.adj.set_adj_bg()
    v.graph_cut(arg['gctype'],lite=True)
    print("Graph Cut Complete")
    return v.labels.v

def dummy_cmd(arg,im,im_gray,im_prev,seed):
    import functools
    v = matsci.gco.Slice(im_gray,seed,bg=True,lightweight=True)
    print("Initialized")
    l = v.new_dummy_label()
    v.data.dilate_all(arg['d'])
    v.data.regions[-1] = matsci.data.dilate(matsci.data.labels_to_edges(v.labels.v),arg['d2'])
    v.data.convert_to_int16()
    def boost_val(n,m):
        m[m==0] = n
        return m
    v.data.regions = map(functools.partial(boost_val,1)
                         ,v.data.regions[:-1]) \
        + [boost_val(0,v.data.regions[-1])]
    # v.data.output_data_term()
    print("Dilated")
    v.graph_cut(arg['gctype'],lite=True)
    print("Graph Cut Complete")
    # import scipy
    # scipy.misc.imsave("d.png",matsci.data.bool_to_uint8(v.labels.v==l))
    if not np.any(v.labels.v==l):
        print('ERROR: no new label')
    v.labels.split_label(l)
    v.labels.clean()
    return v.labels.v

def auto_cmd(arg,im,im_gray,im_prev,seed):
    v = matsci.gco.Slice(im_gray,seed)
    print("Initialized")
    v.data.dilate_auto(v.img,v.labels,arg['d'])
    # v.data.output_data_term()
    print("Dilated")
    v.graph_cut(arg['gctype'])
    print("Graph Cut Complete")
    return v.labels.v

def globalgui_cmd(arg,im,im_gray,im_prev,seed):
    v = matsci.gco.Slice(im_gray,seed)
    print("Initialized")
    v.data.dilate_all(arg['d'])
    print("Dilated")
    v.graph_cut(arg['gctype'])
    v.edit_labels_gui(5)
    # import code; code.interact(local=locals())
    return v.labels.v

def gui_cmd(arg,im,im_gray,im_prev,seed):
    v = matsci.gco.Slice(im_gray,seed)
    print("Initialized")
    v.edit_labels_gui(5)
    return v.labels.v

def skel_cmd(arg,im,im_gray,im_prev,seed):
    v = matsci.gco.Slice(im_gray,seed)
    print("Initialized")
    v.data.dilate(arg['d'])
    v.data.dilate_first(arg['d'])
    print("Dilated")
    v.data.skel(v.orig)
    return v.graph_cut(arg['gctype'])

def log_cmd(arg,im,im_gray,im_prev,seed):
    v = matsci.gco.Slice(im_gray,seed)
    print("Initialized")
    v.data.dilate_first(int(arg['d3']))
    v.data.fit_log(v.img,arg['d'],arg['d2'])
    return v.graph_cut(arg['gctype'])

def gauss_cmd(arg,im,im_gray,im_prev,seed):
    v = matsci.gco.Slice(im_gray,seed)
    print("Initialized")
    # v.dilate_first(arg['d']/10)
    v.data.fit_gaussian(v.img,arg['d'],arg['d2'],arg['d3'])
    return v.graph_cut(arg['gctype'])

def filtergui_cmd(arg,im,im_gray,im_prev,seed):
    """opens gui without doing a cleaning first"""
    v = matsci.gco.Slice(im_gray,seed,lightweight=True)
    v.edit_labels_gui(5)
    return v.graph_cut(arg['gctype'])

def clique_cmd(arg,im,im_gray,im_prev,seed):
    v = matsci.gco.Slice(im_gray,seed)
    v.clique_swap(arg['d'],f=None)
    return v.labels.v

def clique2_cmd(arg,im,im_gray,im_prev,seed):
    v = matsci.gco.Slice(im_gray,seed)
    v.clique_swap(arg['d'],
                  lambda x: max(x.adj.degs(ignore_bg=True, 
                                           ignore=[x.rev_shift(x.center)]),
                                key=lambda y: y[1])[1] > 3)
    return v.labels.v

def clique_compare_cmd(arg,im,im_gray,im_prev,seed):
    v2 = matsci.gco.Slice(im_gray,seed)
    v2.data.dilate_fixed_center(arg['d'], rel_size=0.1, min_size=15, first=True)
    # v2.non_homeomorphic_remove(10,50)
    # v2.data.dilate_all(arg['d'])
    v2.graph_cut(1)
    v = matsci.gco.Slice(im_gray,seed)
    v.clique_swap(arg['d'])
    import matsciskel
    import cv2
    cv2.imwrite('cliquetest.png',matsciskel.draw_on_img(matsciskel.draw_on_img(im,matsciskel.label_to_bmp(v2.labels.v)),matsciskel.label_to_bmp(v.labels.v),color=(0,0,255)))
    cv2.imwrite('cliquetest_global.png',matsciskel.draw_on_img(im,matsciskel.label_to_bmp(v2.labels.v),color=(0,0,255)))
    cv2.imwrite('cliquetest_local.png',matsciskel.draw_on_img(im,matsciskel.label_to_bmp(v.labels.v),color=(0,0,255)))
    return v.labels.v

def clique_compare2_cmd(arg,im,im_gray,im_prev,seed):
    v2 = matsci.gco.Slice(im_gray,seed)
    v2.clique_swap(arg['d'],
                   lambda x: max(x.adj.degs(ignore_bg=True, 
                                            ignore=[x.rev_shift(x.center)]),
                                 key=lambda y: y[1])[1] > 3)
    v = matsci.gco.Slice(im_gray,seed)
    v.clique_swap(arg['d'])
    import matsciskel
    import cv2
    cv2.imwrite('cliquetest.png',matsciskel.draw_on_img(matsciskel.draw_on_img(im,matsciskel.label_to_bmp(v2.labels.v)),matsciskel.label_to_bmp(v.labels.v),color=(0,0,255)))
    cv2.imwrite('cliquetest_global.png',matsciskel.draw_on_img(im,matsciskel.label_to_bmp(v2.labels.v),color=(0,0,255)))
    cv2.imwrite('cliquetest_local.png',matsciskel.draw_on_img(im,matsciskel.label_to_bmp(v.labels.v),color=(0,0,255)))
    return v.labels.v

def compare_cmd(arg,im,im_gray,im_prev,seed):
    v = matsci.gco.Slice(im_gray,seed)
    # v.data.dilate_fixed_center(arg['d'], rel_size=0.1, min_size=15, first=True)
    # v.data.dilate_all(arg['d'])
    # v.graph_cut(1)
    # v.clique_swap(arg['d'])
    # v.non_homeomorphic_remove(arg['d'],arg['d'])
    v.non_homeomorphic_yjunction(arg['d'],r3=7)
    import matsciskel
    import cv2
    cv2.imwrite('cliquetest_local.png',matsciskel.draw_on_img(im,matsciskel.label_to_bmp(v.labels.v),color=(0,0,255)))
    return v.labels.v

def local_stats_cmd(arg,im,im_gray,im_prev,seed):
    import pylab
    v = matsci.gco.Slice(im_gray,seed)
    adj = v.local_adj()
    center = []
    maxdeg = []
    mindeg = []
    for a in adj:
        center += [ np.sort(np.subtract(a.v.sum(axis=0),1))[-1] ]
        maxdeg += [ np.sort(np.subtract(a.v.sum(axis=0),1))[-2] ]
        mindeg += [ np.sort(np.subtract(a.v.sum(axis=0),1))[0] ]
    # pylab.hist(center,align='left',bins=range(1,15))
    # pylab.xlabel('local region degrees')
    # pylab.ylabel('number of regions with degree')
    # pylab.show()
    # pylab.hist(maxdeg,align='left',bins=range(1,15))
    # pylab.xlabel('degree of surrounding grain (max)')
    # pylab.ylabel('number of regions with degree')
    # pylab.show()
    # pylab.hist(mindeg,align='left',bins=range(1,15))
    # pylab.xlabel('degree of surrounding grain (min)')
    # pylab.ylabel('number of regions with degree')
    # pylab.show()
    import cPickle as pickle
    import os.path
    pickle.dump(center,open(os.path.basename(arg['label']).split('.')[0]+'-center.pkl','w'))
    pickle.dump(maxdeg,open(os.path.basename(arg['label']).split('.')[0]+'-max.pkl','w'))
    pickle.dump(mindeg,open(os.path.basename(arg['label']).split('.')[0]+'-min.pkl','w'))
    print('Avg Center Deg: ' + str(np.mean(center)))
    print('Avg Max Deg: ' + str(np.mean(maxdeg)))
    print('Avg Min Deg: ' + str(np.mean(mindeg)))
    return v.labels.v

def color_cmd(arg,im,im_gray,im_prev,labels):
    import scipy
    from scipy.stats import norm
    from scipy.ndimage.filters import gaussian_filter
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from skimage import filter

    norm_const = 100

    # sx = sobel(im, axis=0, mode='constant')
    # sy = sobel(im, axis=1, mode='constant')
    # sob = np.hypot(sx, sy)

    # edges = filter.canny(gaussian_filter(im_gray,4), sigma=1)
    # plt.imshow(edges,cmap = cm.Greys_r)
    # plt.show()

    # edges = scipy.misc.imread('edge/0000-r.png',flatten=True).astype('bool')

    def normalize(img):
        return img.astype('float') / img.max()

    def fit(img,mask):
        return norm.fit(normalize(img)[mask])

    def color_fit(img,mask):
        # these names aren't actually correct, methinks
        r = img[:,:,0]
        g = img[:,:,1]
        b = img[:,:,2]
        return (fit(r,mask),fit(g,mask),fit(b,mask))

    def gauss(mu,sigma):
        n = norm(loc=mu,scale=sigma)
        return n.pdf

    # list of gaussian mu/sigma for each channel for each label
    norms = [ color_fit(im_prev,(labels==l)) for l in range(0,labels.max()+1) ]

    d = [ np.min(np.dstack((gauss(*rn)(normalize(im[:,:,0]))
                            , gauss(*gn)(normalize(im[:,:,1]))
                            , gauss(*bn)(normalize(im[:,:,2]))))
                  ,axis=2)
          for (rn,gn,bn) in norms ]

    m = max([x.max() for x in d])
    d = [(norm_const-(norm_const*x/m)).astype('int16') # - (edges*norm_const).astype('int16')
         for x in d]

    # for x in d:
    #     plt.imshow(x,cmap = cm.Greys_r)
    #     plt.show()

    v = matsci.gco.Slice(im_gray.astype('uint8'),labels)
    v.data = matsci.data.Data()
    v.data.regions = d
    # v.adj.set_adj_all()
    # return v.clique_swap(0)
    return v.graph_cut(arg['gctype'])

