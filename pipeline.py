import matplotlib.pyplot as plt
import pickle
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from helper_functions import *
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space):
    draw_img = np.copy(img)
    hot_windows = []
    #img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]

    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
    else:
        ctrans_tosearch = np.copy(img_tosearch)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                hot_windows.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
                cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)

    return hot_windows, draw_img

class VehicleDetector:
    def __init__(self):
        ### TODO: Tweak these parameters and see how the results change.
        self.color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 13  # HOG orientations
        self.pix_per_cell = 8  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (16, 16)  # Spatial binning dimensions
        self.hist_bins = 16  # Number of histogram bins
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off
        self.heat = None
        self.fmm_cnt = 0
        import os.path
        if os.path.isfile('svc_pickle.p') :
            svc_pickle = pickle.load( open("svc_pickle.p", "rb" ))
            self.svc = svc_pickle["svc"]
            self.X_scaler = svc_pickle["X_scaler"]
        else:
            # Divide up into cars and notcars
            cars = load_images('.\\training\\vehicles')
            notcars = load_images('.\\training\\non-vehicles')

            # Reduce the sample size because
            # The quiz evaluator times out after 13s of CPU time
            sample_size = 8000
            cars = cars[0:sample_size]
            notcars = notcars[0:sample_size]



            car_features = extract_features(cars, color_space=self.color_space,
                                            spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                            orient=self.orient, pix_per_cell=self.pix_per_cell,
                                            cell_per_block=self.cell_per_block,
                                            hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                            hist_feat=self.hist_feat, hog_feat=self.hog_feat)
            notcar_features = extract_features(notcars, color_space=self.color_space,
                                               spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                               orient=self.orient, pix_per_cell=self.pix_per_cell,
                                               cell_per_block=self.cell_per_block,
                                               hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                               hist_feat=self.hist_feat, hog_feat=self.hog_feat)

            X = np.vstack((car_features, notcar_features)).astype(np.float64)
            # Fit a per-column scaler
            self.X_scaler = StandardScaler().fit(X)
            # Apply the scaler to X
            scaled_X = self.X_scaler.transform(X)

            # Define the labels vector
            y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

            # Split up data into randomized training and test sets
            rand_state = np.random.randint(0, 100)
            X_train, X_test, y_train, y_test = train_test_split(
                scaled_X, y, test_size=0.2, random_state=rand_state)

            print('Using:', self.orient, 'orientations', self.pix_per_cell,
                  'pixels per cell and', self.cell_per_block, 'cells per block')
            print('Feature vector length:', len(X_train[0]))
            # Use a linear SVC
            self.svc = LinearSVC()
            # Check the training time for the SVC
            t = time.time()
            self.svc.fit(X_train, y_train)
            t2 = time.time()
            print(round(t2 - t, 2), 'Seconds to train SVC...')
            # Check the score of the SVC
            print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
            # Check the prediction time for a single sample
            t = time.time()
            svc_pickle = {}
            svc_pickle['svc'] = self.svc
            svc_pickle['X_scaler'] = self.X_scaler
            pickle.dump(svc_pickle, open('svc_pickle.p', 'wb'))

    def process_image(self, image):
        orig_image = np.copy(image)
        if image.dtype is not np.float32:
            image = image.astype(np.float32) / 255
        hot_windows = []
        if self.heat is None:
            self.heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        ystart = 300
        ystop = 600
        scale = 1.5
        scale_windows, out_img = find_cars(image, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins, self.color_space)
        hot_windows += scale_windows

        #plt.imshow(out_img)
        #plt.show()

        ystart = 325
        ystop = 656
        scale = 1.75
        scale_windows, out_img = find_cars(image, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins, self.color_space)
        hot_windows += scale_windows

        #plt.imshow(out_img)
        #plt.show()

        ystart = 375
        ystop = 700
        scale = 3
        scale_windows, out_img = find_cars(image, ystart, ystop, scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins, self.color_space)
        hot_windows += scale_windows

        #plt.imshow(out_img)
        #plt.show()
        self.heat *= 0.8
        #self.heat = self.heat.clip(min=0, max=0.5)
        # Add heat to each box in box list
        self.heat = add_heat(self.heat, hot_windows)

        # Apply threshold to help remove false positives
        curr_heat = apply_threshold(self.heat, 15.1)

        # Visualize the heatmap when displaying
        heatmap = np.clip(curr_heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(orig_image, labels)


        return draw_img

vehicle_detector = VehicleDetector()
#image = mpimg.imread('.\\test_images\\test6.jpg')
#draw_img = vehicle_detector.process_image(image)
#plt.imshow(draw_img)
#plt.show()
clip1 = VideoFileClip("./project_video.mp4")
res_video = clip1.fl_image(vehicle_detector.process_image)
res_video.write_videofile("./project_video_res.mp4", audio=False)