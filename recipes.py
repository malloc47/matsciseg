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
