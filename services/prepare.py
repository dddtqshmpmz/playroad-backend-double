#-*-coding:utf-8-*-

"""
@Class: 人工智能基础与实践
@FileName: 决策树体验
@Author: 上海交通大学
@LastUpdate: 2018.7.13
"""

import cv2 
import glob
import os
import numpy as np 
import pandas as pd

def _get_data(img_dir):
    dfs = []
    for train_file in glob.glob(os.path.join(img_dir, '*/GT-*.csv')):
        #print train_file
        folder = '/'.join(train_file.split('\\')[:-1])
        #print folder
        df = pd.read_csv(train_file, sep=';')
        df['Filename'] = df['Filename'].apply(lambda x: os.path.join( folder, x))
        #print df['Filename'][0]
        dfs.append(df)

    train_df = pd.concat(dfs, ignore_index=True)

    # print train_df.head()
    n_classes = np.unique(train_df['ClassId']).size
    print('Number of training images : {:>5}'.format(train_df.shape[0]))
    print('Number of classes         : {:>5}'.format(n_classes))
    
    return train_df

def _get_features(data, feature=None, cut_roi=False, test_split=0.2, seed=113):
    """Loads the GTSRB dataset
        This function loads the German Traffic Sign Recognition Benchmark
        (GTSRB), performs feature extraction, and partitions the data into
        mutually exclusive training and test sets.
     
        :param feature:      which feature to extract: None, "gray", "rgb",
                        "hsv", or "hog"
        :param cut_roi:      flag whether to remove regions surrounding the
                        actual traffic sign (True) or not (False)
        :param test_split:   fraction of samples to reserve for the test set
        :param plot_samples: flag whether to plot samples (True) or not (False)
        :param seed:         which random seed to use
        :returns:            (X_train, y_train), (X_test, y_test)
    """

    # read all training samples and corresponding class labels
    X = []  # data 
    labels = []  # corresponding labels
    for c in range(len(data)):
        im = cv2.imread(data['Filename'].values[c])
        # first column is filename
        # im = cv2.imread(data['Filename'].value[c])

        # remove regions surrounding the actual traffic sign
        if cut_roi:
            im = im[np.int(data['Roi.X1'].values[c]):np.int(data['Roi.X2'].values[c]),\
                    np.int(data['Roi.Y1'].values[c]):np.int(data['Roi.Y2'].values[c]), :]

        X.append(im)
        labels.append(data['ClassId'].values[c])
    
    # perform feature extraction
    X = extract_feature(X, feature)

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)

    X_train = X[:int(len(X)*(1-test_split))]
    y_train = labels[:int(len(X)*(1-test_split))]

    X_test = X[int(len(X)*(1-test_split)):]
    y_test = labels[int(len(X)*(1-test_split)):]
    
    return (X_train, y_train), (X_test, y_test)

def extract_feature(X, feature):
    """Performs feature extraction
    :param X:       data (rows=images, cols=pixels)
    :param feature: which feature to extract 
        - None:   no feature is extracted
        - "gray": grayscale features
        - "rgb":  RGB features 
        - "hsv":  HSV features 
        - "hog":  HOG features 
        :returns:       X (rows=samples, cols=features)
    """

    # transform color space 
    if feature == 'gray':
        X = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in X]
    elif feature == 'hsv':
        X = [cv2.cvtColor(x, cv2.COLOR_BGR2HSV) for x in X]

    # operate on smaller image 
    small_size = (32, 32)
    X = [cv2.resize(x, small_size) for x in X]

    # extract features 
    if feature == 'hog':
        # histogram of gradients
        block_size = (small_size[0] // 2, small_size[1] // 2)
        block_stride = (small_size[0] // 4, small_size[1] // 4)
        cell_size = block_stride
        num_bins = 9
        hog = cv2.HOGDescriptor(small_size, block_size, block_stride, cell_size, num_bins)
        X = [hog.compute(x) for x in X]

    elif feature is not None:
        # normalize all intensities to be between 0 and 1
        X = np.array(X).astype(np.float32) / 255

        # subtract mean 
        X = [x - np.mean(x) for x in X]

    X = [x.flatten() for x in X]
    
    return X

def prepare_data(feature_type):
    img_dir = './data/train_data' 
    train_data = './feature/train/train_data.npy'
    train_label = './feature/train/train_label.npy'
    val_data = './feature/val/val_data.npy'
    val_label = './feature/val/val_label.npy'
        
    types = ['hog', 'gray', 'hsv', 'rgb']
    if feature_type not in types:
        print('Error: unknown feature type!')
        exit(0)
    else:
        train_df = _get_data(img_dir)
        (x_train, y_train), (x_valid, y_valid) = _get_features(train_df, feature_type, cut_roi=False, test_split=0.2, seed=113)
        
        np.save(train_data, x_train)
        np.save(train_label, y_train)
        np.save(val_data, x_valid)
        np.save(val_label, y_valid)

        data = []
        data.append(x_train)
        data.append(y_train)
        data.append(x_valid)
        data.append(y_valid)
        return data