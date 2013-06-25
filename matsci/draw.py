import numpy as np

def display(im):
    import cv,cv2
    cv2.namedWindow("tmp",cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow("tmp",im)
    cv2.waitKey(0)

def label_to_bmp(labels):
    grad = np.gradient(labels)
    seg = np.maximum(abs(grad[0]),abs(grad[1]))
    return seg

def draw_on_img(img,bmp,color=(255,0,0)):
    out = img.copy()
    out[np.nonzero(bmp>0)] = color
    return out

# def alpha_blend(img,bmp,color=(255,0,0,0.5)):
#     out = img.copy()
#     out[np.nonzero(bmp>0)] = np.add(out[np.nonzero(bmp>0)],tuple([c*color[-1] for c in color[:-1]]))
#     return out

def alpha_blend(img,bmp,color=(255,0,0,0.5)):
    out = img.copy().astype('int16')
    out[np.nonzero(bmp>0)] = np.add(out[np.nonzero(bmp>0)],tuple([c*color[-1] for c in color[:-1]]))
    return np.clip(out,0,255).astype('uint8')

jet_colors = [(0,0,143), (0,0,159), (0,0,175), (0,0,191), (0,0,207), (0,0,223), (0,0,239), (0,0,255), (0,16,255), (0,32,255), (0,48,255), (0,64,255), (0,80,255), (0,96,255), (0,112,255), (0,128,255), (0,143,255), (0,159,255), (0,175,255), (0,191,255), (0,207,255), (0,223,255), (0,239,255), (0,255,255), (16,255,239), (32,255,223), (48,255,207), (64,255,191), (80,255,175), (96,255,159), (112,255,143), (128,255,128), (143,255,112), (159,255,96), (175,255,80), (191,255,64), (207,255,48), (223,255,32), (239,255,16), (255,255,0), (255,239,0), (255,223,0), (255,207,0), (255,191,0), (255,175,0), (255,159,0), (255,143,0), (255,128,0), (255,112,0), (255,96,0), (255,80,0), (255,64,0), (255,48,0), (255,32,0), (255,16,0), (255,0,0), (239,0,0), (223,0,0), (207,0,0), (191,0,0), (175,0,0), (159,0,0), (143,0,0), (128,0,0), (0,0,159), (0,0,175), (0,0,191), (0,0,207), (0,0,223), (0,0,239), (0,0,255), (0,16,255), (0,32,255), (0,48,255), (0,64,255), (0,80,255), (0,96,255), (0,112,255), (0,128,255), (0,143,255), (0,159,255), (0,175,255), (0,191,255), (0,207,255), (0,223,255), (0,239,255), (0,255,255), (16,255,239), (32,255,223), (48,255,207), (64,255,191), (80,255,175), (96,255,159), (112,255,143), (128,255,128), (143,255,112), (159,255,96), (175,255,80), (191,255,64), (207,255,48), (223,255,32), (239,255,16), (255,255,0), (255,239,0), (255,223,0), (255,207,0), (255,191,0), (255,175,0), (255,159,0), (255,143,0), (255,128,0), (255,112,0), (255,96,0), (255,80,0), (255,64,0), (255,48,0), (255,32,0), (255,16,0), (255,0,0), (239,0,0), (223,0,0), (207,0,0), (191,0,0), (175,0,0), (159,0,0), (143,0,0), (128,0,0)]
randmap = [54,6,46,17,35,63,1,23,28,27,49,2,13,21,14,39,30,47,45,4,26,5,57,61,55,52,32,34,41,18,12,25,53,56,22,19,48,9,37,8,60,36,64,16,3,58,44,15,7,10,20,24,40,11,43,50,38,51,59,29,62,31,33,42,54,6,46,17,35,63,1,23,28,27,49,2,13,21,14,39,30,47,45,4,26,5,57,61,55,52,32,34,41,18,12,25,53,56,22,19,48,9,37,8,60,36,64,16,3,58,44,15,7,10,20,24,40,11,43,50,38,51,59,29,62,31,33,42]


def color_jet(img,labels,alpha=0.5):
    output = img.copy()
    for l in range(labels.min(),labels.max()):
        output[labels==(l%128)] = jet_colors[randmap[l%128]]
    output = (output*alpha+img*(1-alpha)).astype('uint8')
    return output


def grey_to_rgb(im):
    return np.repeat(im,3,axis=1).reshape(im.shape+(3,))

def salient(label,im):
    # label = matsci.io.read_labels('seq1/global-20/90/image0099.label')
    # _,im = matsci.io.read_img('seq1/img/image0099.png')
    import matsci.io
    from skimage.transform import probabilistic_hough
    from skimage.morphology import remove_small_objects
    from skimage.filter import canny, threshold_adaptive
    from scipy.ndimage.morphology import distance_transform_edt
    # lines = probabilistic_hough(im_grey, threshold=50, line_length=5, line_gap=3)
    im_edges = canny(im.astype('float')/255, sigma=1.0, low_threshold=0.1, high_threshold=0.2, mask=None)
    im_edges = remove_small_objects(im_edges,50,2)
    # edges = threshold_adaptive(im,101,method='gaussian')
    # lines = probabilistic_hough(edges, threshold=50, line_length=10, line_gap=1)

    seg = label_to_bmp(label)
    dt = distance_transform_edt(np.logical_not(seg))
    min_d, max_d = 5, 50

    im_edges[dt<min_d] = 0
    im_edges[dt>max_d] = 0

    # future pipeline:
    # - connected components
    # - elipse fitting
    # - learn ellipse parameters
    # - classify components

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    plt.figure()
    plt.imshow(im_edges, cmap = cm.Greys_r)
    # for line in lines:
    #     p0, p1 = line
    #     plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    plt.show()
