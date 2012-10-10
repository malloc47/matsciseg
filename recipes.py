import matsci.gco

def global_cmd(arg,im,im_gray,seed):
    v = matsci.gco.Slice(im_gray,seed)
    print("Initialized")
    v.data.dilate_all(arg['d'])
    v.data.output_data_term()
    print("Dilated")
    # v.set_adj_label_all(0);
    # print("Adjacent")
    v.graph_cut(arg['gctype'])
    print("Graph Cut Complete")
    # import code; code.interact(local=locals())
    return v.labels.v

def auto_cmd(arg,im,im_gray,seed):
    v = matsci.gco.Slice(im_gray,seed)
    print("Initialized")
    v.data.dilate_auto(v.img,v.labels,arg['d'])
    v.data.output_data_term()
    print("Dilated")
    v.graph_cut(arg['gctype'])
    print("Graph Cut Complete")
    return v.labels.v

def globalgui_cmd(arg,im,im_gray,seed):
    v = matsci.gco.Slice(im_gray,seed)
    print("Initialized")
    v.data.dilate_all(arg['d'])
    print("Dilated")
    v.graph_cut(arg['gctype'])
    v.edit_labels_gui(5)
    # import code; code.interact(local=locals())
    return v.labels.v

def gui_cmd(arg,im,im_gray,seed):
    v = matsci.gco.Slice(im_gray,seed)
    print("Initialized")
    v.edit_labels_gui(5)
    return v.labels.v

def skel_cmd(arg,im,im_gray,seed):
    v = matsci.gco.Slice(im_gray,seed)
    print("Initialized")
    v.data.dilate(arg['d'])
    v.data.dilate_first(arg['d'])
    print("Dilated")
    v.data.skel(v.orig)
    return v.graph_cut(arg['gctype'])

def gauss_cmd(arg,im,im_gray,seed):
    v = matsci.gco.Slice(im_gray,seed)
    print("Initialized")
    # v.dilate_first(arg['d']/10)
    v.data.fit_gaussian(v.img,arg['d'],arg['d2'],arg['d3'])
    return v.graph_cut(arg['gctype'])

def filtergui_cmd(arg,im,im_gray,seed):
    """opens gui without doing a cleaning first"""
    v = matsci.gco.Slice(im_gray,seed,lightweight=True)
    v.edit_labels_gui(5)
    return v.graph_cut(arg['gctype'])

def clique_cmd(arg,im,im_gray,seed):
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

def compare_cmd(arg,im,im_gray,seed):
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
