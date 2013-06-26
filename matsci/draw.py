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

def extract_features(im,label):
    from scipy.ndimage.morphology import distance_transform_edt
    from skimage.morphology import remove_small_objects, binary_dilation, disk
    from skimage.morphology import label as label_im
    from skimage.filter import canny
    # from skimage.filter import threshold_adaptive

    # lines = probabilistic_hough(im_grey, threshold=50, line_length=5, line_gap=3)

    # edges = threshold_adaptive(im,101,method='gaussian')
    # lines = probabilistic_hough(edges, threshold=50, line_length=10, line_gap=1)

    im_edges = canny(im.astype('float')/255, sigma=1.0, low_threshold=0.1, high_threshold=0.2, mask=None)
    im_edges = remove_small_objects(im_edges,40,2)

    seg = label_to_bmp(label)
    dt = distance_transform_edt(np.logical_not(seg))
    min_d, max_d = 5, 50

    im_edges[dt<min_d] = 0
    im_edges[dt>max_d] = 0

    im_edges = binary_dilation(im_edges,disk(5))

    return label_im(im_edges, neighbors=8, background=0)

# export_features((('seq1/img/image0091.png','seq1/global-20/90/image0091.label'),
#                  ('seq1/img/image0092.png','seq1/global-20/90/image0092.label'),
#                  ('seq1/img/image0093.png','seq1/global-20/90/image0093.label'),
#                  ('seq1/img/image0094.png','seq1/global-20/90/image0094.label'),
#                  ('seq1/img/image0095.png','seq1/global-20/90/image0095.label'),
#                  ('seq1/img/image0096.png','seq1/global-20/90/image0096.label'),
#                  ('seq1/img/image0097.png','seq1/global-20/90/image0097.label'),
#                  ('seq1/img/image0098.png','seq1/global-20/90/image0098.label'),
#                  ('seq1/img/image0099.png','seq1/global-20/90/image0099.label'),
#                  ('seq1/img/image0100.png','seq1/global-20/90/image0100.label')))
def export_features(im_label_pairs):
    import os
    import matsci.io
    from skimage.measure import regionprops
    win = 35
    try:
        os.mkdir('features')
    except:
        pass
    idx = 0
    feature_file = 'features/features'
    with open(feature_file, 'a') as f:
        f.write('[\n')
    for im_path, label_path in im_label_pairs:
        label = matsci.io.read_labels(label_path)
        _,im = matsci.io.read_img(im_path)
        salient_regions = extract_features(im,label)
        props = regionprops(salient_regions,
                            intensity_image=im,
                            properties=['Area','BoundingBox','EquivDiameter', 
                                        'MajorAxisLength','MinorAxisLength',
                                        'Eccentricity','EquivDiameter',
                                        'MaxIntensity','MeanIntensity'])
        for p in props:
            y0, x0, y1, x1 = p['BoundingBox']
            ymin = max(min(y0,y1)-win,0)
            ymax = min(max(y0,y1)+win,im.shape[0]-1)
            xmin = max(min(x0,x1)-win,0)
            xmax = min(max(x0,x1)+win,im.shape[1]-1)
            feature_vector = (idx,
                              p['Area'], 
                              p['EquivDiameter'],
                              p['MajorAxisLength'],
                              p['MinorAxisLength'],
                              p['Eccentricity'],
                              p['EquivDiameter'],
                              p['MaxIntensity'],
                              p['MeanIntensity'],
                              0) # last is class to be changed later
            img = im.copy()
            mask = np.zeros(im.shape,dtype=bool)
            mask[y0:y1,x0:x1] = True
            img[np.logical_not(mask)] /= 4
            matsci.io.write_img(img[ymin:ymax,xmin:xmax],
                                'features/'+str(idx)+'.png')
            with open(feature_file, 'a') as f:
                f.write(str(feature_vector)+',\n')
            idx += 1
    with open(feature_file, 'a') as f:
        f.write(']\n')

def train_classifier(feature_path):
    from sklearn import svm
    from sklearn.externals import joblib

    with open(feature_path,'r') as f:
        raw_features = eval(f.read())

    features = [f[1:-1] for f in raw_features]
    targets = [f[-1] for f in raw_features]

    classifier = svm.SVC()
    classifier.fit(features,targets)

    joblib.dump(classifier,'classifiers/salient')

    return classifier

classifier = None
            
def salient(im,label,use_classifier=True):
    import matsci.io, matsci.draw
    # label = matsci.io.read_labels('seq1/global-20/90/image0099.label')
    # _,im = matsci.io.read_img('seq1/img/image0099.png')
    # from skimage.transform import probabilistic_hough

    from skimage.measure import regionprops
    from skimage.draw import polygon, line

    # future pipeline:
    # - connected components
    # - elipse fitting
    # - learn ellipse parameters
    # - classify components

    salient_regions = extract_features(im,label)

    props = regionprops(salient_regions,
                        intensity_image=im,
                        properties=['Area','BoundingBox','EquivDiameter', 
                                        'MajorAxisLength','MinorAxisLength',
                                        'Eccentricity','EquivDiameter',
                                        'MaxIntensity','MeanIntensity'])

    mask = np.zeros(im.shape,dtype=bool)

    if use_classifier:
        global classifier

        if not classifier:
            from sklearn.externals import joblib
            classifier = joblib.load('classifiers/salient')

        props = [p for p in props 
                    if classifier.predict((p['Area'], 
                                           p['EquivDiameter'],
                                           p['MajorAxisLength'],
                                           p['MinorAxisLength'],
                                           p['Eccentricity'],
                                           p['EquivDiameter'],
                                           p['MaxIntensity'],
                                           p['MeanIntensity']))[0]]

    for p in props:
        y0, x0, y1, x1 = p['BoundingBox']
        # x0,y0 -> x0,y1 -> x1,y1 -> x1 y0 -> x0,y0
        rr, cc = polygon(y=np.array((y0,y1,y1,y0,y0)),
                         x=np.array((x0,x0,x1,x1,x0)))
        # rr, cc = ellipse(cy=p['Centroid'][0],
        #                  cx=p['Centroid'][1],
        #                  xradius=p['MajorAxisLength'],
        #                  yradius=p['MinorAxisLength'])
        mask[rr,cc] = True

    out = im.copy()

    out[np.logical_not(mask)] /= 8

    blue = out.copy()

    # draw blue boxes
    for p in props:
        y0, x0, y1, x1 = p['BoundingBox']
        y0 = min(y0,im.shape[0]-1)
        y1 = min(y1,im.shape[0]-1)
        x0 = min(x0,im.shape[1]-1)
        x1 = min(x1,im.shape[1]-1)
        # x0,y0 -> x0,y1 -> x1,y1 -> x1 y0 -> x0,y0
        rr, cc = line(y0,x0,y1,x0)
        blue[rr,cc] = 255
        rr, cc = line(y1,x0,y1,x1)
        blue[rr,cc] = 255
        rr, cc = line(y1,x1,y0,x1)
        blue[rr,cc] = 255
        rr, cc = line(y0,x1,y0,x0)
        blue[rr,cc] = 255

    out = np.dstack((out,out,blue))
    out = matsci.draw.draw_on_img(out,label_to_bmp(label))

    # import matplotlib.pyplot as plt
    # import matplotlib.cm as cm
    # plt.figure()
    # plt.imshow(im_edges, cmap = cm.Greys_r)
    # # for line in lines:
    # #     p0, p1 = line
    # #     plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
    # plt.show()

    return out
