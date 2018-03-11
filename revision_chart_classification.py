from matplotlib import pyplot as plt
from multiprocessing import Pool
from scipy.stats import mode
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import cv2
import numpy as np
import os
import random
import sklearn
import time


K = 200
# path to directory with raster chart images
revision_path = '/Users/duanp/revision/revision_charts/'
types_dict = {
#    "AreaGraph": 1,
    "BarGraph": 2,
    "LineGraph": 3,
    "Map": 4,
 #   "ParetoChart": 5,
 #   "PieChart": 6,
#    "RadarPlot": 7,
    "ScatterGraph": 8,
    "Table": 9
#    "VennDiagram": 10
}

reverse_types_dict = {}
count = {}
for key, val in types_dict.items():
    reverse_types_dict[val] = key
    count[key] = 0


def normalize_img(img_path):
    img = cv2.imread(img_path, 0)
    h,w = img.shape[:2]
    max_dim = max(h,w)
    ratio = 128.0/max_dim
    top = 0
    bottom = 0
    left = 0
    right = 0
    if w == max_dim:
        w = 128
        h = int(h*ratio)
        padding = int((128 - h)/2.0)
        top = padding
        bottom = padding
    else:
        h = 128
        w = int(w*ratio)
        padding = int((128 - w)/2.0)
        left = padding
        right = padding
    img1 = cv2.resize(img, (w, h))
    h_borders = np.concatenate((img1[0,:], img1[h-1,:]))
    v_borders = np.concatenate((img1[:,w-1], img1[:,0]))
    color = mode(np.concatenate((h_borders, v_borders)))[0][0]
    output = cv2.copyMakeBorder(img1, top, bottom, right, left, cv2.BORDER_CONSTANT, value=[int(color)])
    return cv2.copyMakeBorder(output, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[int(color)])

def classify(img_paths):
    start = time.time()
    normalized_imgs = []
    labels = []
    start = time.time()
    for img_path, label, url, chart_num in img_paths:
        try:
            normalized_imgs.append(normalize_img(img_path))
            labels.append(label)
            count[reverse_types_dict[label]] += 1
        except Exception as e:
            print e
            wrong_urls.write(url)
            if url.rstrip('\n').lower()[-4:] == ".gif":
                gifs.write(chart_num + '\t' + url)
            print url[-4:]
            pass
    wrong_urls.close()
    gifs.close()
    print count
    print len(normalized_imgs)
    print "image normalization"
    print time.time() - start
    start = time.time()
    patches = get_patches_overlap(normalized_imgs)
    print "codebook patch extraction"
    print time.time() - start
    start = time.time()
    patches = patch_standardization(patches)
    print "patch standardization"
    print time.time() - start
    start = time.time()
    codebook = k_means_clustering(patches)
    print "k means"
    print time.time() - start
    start = time.time()
    feature_vectors = np.array([get_img_feature_vector(normalized_img, codebook) for normalized_img in normalized_imgs])
    print "get feature vectors"
    print time.time() - start

def get_codebook(normalized_imgs):
    start = time.time()
    patches = get_patches_overlap(normalized_imgs)
    print "codebook patch extraction"
    print time.time() - start
    start = time.time()
    patches = patch_standardization(patches, 0)
    return k_means_clustering(patches)

def get_codebook_patch(img_paths):
    normalized_imgs = []
    labels = []
    start = time.time()
    for img_path, label, url in img_paths:
        try:
            normalized_imgs.append(normalize_img(img_path))
            labels.append(label)
            count[reverse_types_dict[label]] += 1
        except Exception as e:
            print e
            print img_path
            pass
    patches = get_patches_overlap(normalized_imgs)
    patches = patch_standardization(patches)
    return k_means_clustering(patches)

def normalize_testing_vectors(test_vecs, mean, std):
    return (test_vecs - mean)/std

def cross_validation(chart_paths, new_labels):
    correct = {}
    total = {}
    normalized_imgs = []
    labels = []

    # Step 1: Image normalization
    for i in range(len(chart_paths)):
        try:
            normalized_imgs.append(normalize_img(chart_paths[i]))
            labels.append(new_labels[i])
        except Exception as e:
            print e
            print chart_paths[i]
            pass
    print len(normalized_imgs)
    skf = StratifiedKFold(labels, n_folds=5)
    for train_index, test_index in skf:
        training_charts = [normalized_imgs[t] for t in train_index]
        training_labels = [labels[t] for t in train_index]
        testing_charts = [normalized_imgs[t] for t in test_index]
        testing_labels = [labels[t] for t in test_index]
        # Steps 2 - 5: Patch Extraction, Standardization, and Clustering for Codebook
        codebook = get_codebook(normalized_imgs)
        print "finished codebook"
        # Steps 5 - 6: Feature Vector Formulation
        training_vecs = np.array([get_img_feature_vector((training_chart, codebook)) for training_chart in training_charts])
        training_mean = np.mean(training_vecs)
        training_std = np.sqrt(np.var(training_vecs) + 0.01)
        testing_vecs = np.array([get_img_feature_vector((testing_chart, codebook)) for testing_chart in testing_charts])
        print "finished feature vectors"
        # Step 7: Classification
        clf = svm.SVC(kernel='poly', degree=2, tol=1e-4, gamma=0.02)
        clf.fit(training_vecs, training_labels)
        for i in range(len(testing_labels)):
            label = testing_labels[i]
            vector = testing_vecs[i]
            if total.get(label):
                total[label] += 1
            else:
                total[label] = 1
                correct[label] = 0
            if np.asscalar(clf.predict(vector)) == label:
                correct[label] += 1
        for k in correct.keys():
            print reverse_types_dict[k]
            print str(correct[k]) + "/" + str(total[k])
        print "average"
        print 1.0*sum(correct.values())/sum(total.values())
        return

def cross_validation1(chart_paths, urls):
    correct = {}
    total = {}
    normalized_imgs = []
    labels = []
    for img_path, label in chart_paths:
        try:
            normalized_imgs.append(normalize_img(img_path))
            labels.append(label)
            count[reverse_types_dict[label]] += 1
        except Exception as e:
            print e
            print img_path
            pass
    print len(normalized_imgs)
    print count
    skf = StratifiedKFold(labels, n_folds=5)
    for train_index, test_index in skf:
        training_charts = [normalized_imgs[t] for t in train_index]
        training_labels = [labels[t] for t in train_index]
        testing_charts = [normalized_imgs[t] for t in test_index]
        testing_labels = [labels[t] for t in test_index]
        codebook = get_codebook(training_charts)
        patch_count = 1
        for center in codebook.cluster_centers_:
            cv2.imwrite(codebook_dir + str(patch_count) + ".png", center.reshape((6,6)))
            patch_count += 1
        print "finished codebook"
        p = Pool(5)
        training_vecs_input = [(training_chart, codebook) for training_chart in training_charts]
        training_vecs = np.array(p.map(get_img_feature_vector, training_vecs_input))
        p.close()
        p.join()
        testing_vecs = np.array([get_img_feature_vector((testing_chart, codebook)) for testing_chart in testing_charts])
        print "finished feature vectors"
        clf = svm.LinearSVC(kernel='poly', degree=2, tol=1e-4, gamma=0.02)
        clf.fit(training_vecs, training_labels)
        for i in range(len(testing_labels)):
            label = testing_labels[i]
            vector = testing_vecs[i]
            if total.get(label):
                total[label] += 1
            else:
                total[label] = 1
                correct[label] = 0
            if np.asscalar(clf.predict(vector)) == label:
                correct[label] += 1
        for k in correct.keys():
            print reverse_types_dict[k]
            print str(correct[k]) + "/" + str(total[k])
        print "average"
        print 1.0*sum(correct.values())/sum(total.values())
        return

def get_patches_overlap(imgs):
    patches = []
    for img in imgs:
        img_patches = []
        seen = set([])
        choices = [(i,j) for i in range(0, 122) for j in range(0, 122)]
        while len(img_patches) < 100:
            x,y = random.choice(choices)
            if (x,y) not in seen:
                seen.add((x,y))
                patch = np.array(img[x:x+6, y:y+6])
                max_pixel = np.max(patch)
                var = np.var(patch)
                if var > 38:
                    img_patches.append(patch)
        patches.extend(img_patches)
    return patches

def normalize_patch(patch, factor):
    std = np.sqrt(np.var(patch) + factor) #revision code
    mean = np.mean(patch)
    patch = (patch - mean)/std
    patch_array = zca_whiten(np.array([patch]))
    return patch_array[0]



def get_img_feature_vector(img_codebook):
    img, codebook = img_codebook
    patches = []
    feature_vector = []
    midpoint1 = 58
    midpoint2 = 64
    for x_start, x_end, y_start, y_end in [(0, midpoint1, 0, midpoint1), (0, midpoint1, midpoint2, 122), (midpoint2, 122, 0, midpoint1), (midpoint2, 122, midpoint2, 122)]:
        for i in range(x_start,x_end):
            for j in range(y_start, y_end):
                patches.append(img[i:i+6, j:j+6])
    patches = patch_standardization(patches, 0.1)
    for start in [0, 4096, 8192, 12288]:
        quad = patches[start:start+4096]
        classification = codebook.predict(quad)
        histogram = np.zeros(K)
        for cluster in classification:
            histogram[cluster] += 1
        feature_vector.extend(histogram)
    return np.array(feature_vector)


def get_quad_histogram(input_arg):
    x_start, x_end, y_start, y_end, codebook, img, x = input_arg
    quad = []
    for i in range(x_start,x_end):
        for j in range(y_start, y_end):
            quad.append(normalize_patch(img[i-3:i+3, j-3:j+3].flatten()))
    classification = codebook.predict(quad)
    histogram = np.zeros(K)
    for cluster in classification:
        histogram[cluster] += 1
    return (x, histogram)


def get_patches_no_overlap(imgs):
    patches = []
    for img in imgs:
        choices = [(i,j) for i in np.arange(0, 128 - 6, 6) for j in np.arange(0, 128 - 6, 6)]
        for x,y in choices:
            patch = img[x:x+6, y:y+6]
            max_pixel = np.max(patch)
            var = np.var(patch)
            if var >= 0.1*max_pixel and var > 0 and max_pixel > 0:
                patches.append(patch)
        patches.extend(random.sample(patches, min(len(patches), 100)))
    return patches

def patch_standardization(patches, factor):
    new_patches = []
    for patch in patches:
        std = np.sqrt(np.var(patch) + factor) #revision code
        mean = np.mean(patch)
        patch = (patch - mean)/std
        new_patches.append(patch.flatten())
    return zca_whiten(np.array(new_patches))

def k_means_clustering(patches):
    clusters = KMeans(n_clusters=K)
    patches = np.float32(patches)
    clusters.fit(patches)
    return clusters

def zca_whiten(X):
    """
    Applies ZCA whitening to the data (X)
    http://xcorr.net/2011/05/27/whiten-a-matrix-matlab-code/

    X: numpy 2d array
        input data, rows are data points, columns are features

    Returns: ZCA whitened 2d array
    """
    assert(X.ndim == 2)
    EPS = 0.1

    #   covariance matrix
    cov = np.dot(X.T, X)
    #   d = (lambda1, lambda2, ..., lambdaN)
    d, E = np.linalg.eigh(cov)
    #   D = diag(d) ^ (-1/2)
    D = np.diag(1. / np.sqrt(d + EPS))
    #   W_zca = E * D * E.T
    W = np.dot(np.dot(E, D), E.T)

    X_white = np.dot(X, W)

    return X_white

remove_mean = True
hard_beta = True
beta = 10.0
gamma = 0.01

def contrast_normalize(patches):
    X = patches
    if X.ndim != 2:
        raise TypeError('contrast_normalize requires flat patches')
    if remove_mean:
        xm = X.mean(1)
    else:
        xm = X[:,0] * 0
    Xc = X - xm[:, None]
    l2 = (Xc * Xc).sum(axis=1)
    if hard_beta:
        div2 = np.maximum(l2, beta)
    else:
        div2 = l2 + beta
    X = Xc / np.sqrt(div2[:, None])
    return X

def ZCA_whiten(patches):
    # -- ZCA whitening (with band-pass)

    # Algorithm from Coates' sc_vq_demo.m

    X = patches.reshape(len(patches), -1).astype('float64')

    X = contrast_normalize(X)
    print 'patch_whitening_filterbank_X starting ZCA'
    M = X.mean(0)
    _std = np.std(X)
    Xm = X - M
    assert Xm.shape == X.shape
    print 'patch_whitening_filterbank_X starting ZCA: dot', Xm.shape
    C = np.dot(Xm.T, Xm) / (Xm.shape[0] - 1)
    print 'patch_whitening_filterbank_X starting ZCA: eigh'
    D, V = np.linalg.eigh(C)
    print 'patch_whitening_filterbank_X starting ZCA: dot', V.shape
    P = np.dot(np.sqrt(1.0 / (D + gamma)) * V, V.T)
    assert M.ndim == 1
    return M, P, X

def main():
    chart_paths = []
    labels = []
    for chart_type in types_dict.keys():
        subdirectory = revision_path + chart_type + "/"
        for file_path in os.listdir(subdirectory):
            chart_paths.append(subdirectory + file_path)
            labels.append(types_dict[chart_type])
    cross_validation(chart_paths, labels)

start = time.time()
main()
print "runtime"
print time.time() - start


