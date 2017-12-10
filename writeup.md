## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README
### Structure

File 'pipeline.py' contains the class VehicleDetector. Its constructor contains the training of an SVM classifier on Histogram of Oriented Gradients (HOG), spatial and color histogram features. The file 'helper_functions.py' contains several helper methods for feature extraction, training and classification.
The classifier's parameters are cached in the file 'svc_pickle.p'. If any hyperparameters for the classifier or the training data is changed, this file needs to be deleted to trigger a re-training of the classifier.
The method process_image applies the classifier to an input image. It also performs tracking, so when it should be applied to a new sequence, a new instance of VehicleDetector should be created.

### Histogram of Oriented Gradients (HOG)

The code for this step is contained in the helper function 'get_hog_features' called by extract_features (for training) and find_car (for classification).

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 1. HOG parameters.

For orientations values around 10 lead to good results. Less orientations led to worse results. I chose 13 channels because this value seemed to be a good compromise between performance and resource requirements. I used all hog-channels as using only a subset didn't yield good results. In terms of color-spaces, YCrCb and YUV led to good results, in general, all color spaces with a separate luminance or brightness channel seemed to perform better.


#### 2. Linear SVM Classifier

I trained a linear SVM using the HOG features as described above as well as 16 color histogram bins and 16x16 spatial features.

### Sliding Window Search

#### 1. Scales

I analized a bunch of test images to find out at which position in the image cars appear and how big the appear at certain positions.
I finally came up with five different scales in different y-coordinate-intervals:
- Scale 1.0 at y from 360 to 480
- Scale 1.25 at y from 380 to 500
- Scale 1.5 at y from 390 t0 520
- Scale 2.0 at y from 390 to 600
- Scale 2.5 at y from 390 to 650

#### 2. HOG Sub-sampling

I used HOG sub-sampling to speed up feature computation. The hog features are computed on the whole image (in the pre-defined y-range) in lines 137-139 of function find_cars and the appropriately aggregated within the following two for-loops where the features for the individual windows are computed.

![alt text][image3]

![alt text][image4]
---

### Video Implementation

#### 1. Result
Here's a [link to my video result](./project_video_res.mp4)


#### 2. False-positive filter and tracking

I decided to use a heat-map approach for false-positive filtering and tracking. First of all, a heat-map is computed by aggregating the all detections from all windows. The heat-map is then thresholded and segmented to yield actual detections.
The heat-map is also used for the detection of he upcoming frames: The previous heat-map is multiplied by 0.8 (IIR-averaging) and then the new heat-map is added to it. This way, one-time outliers in a few individual frames are suppressed whereas continuous detections aggregate and increase the heat-map values around the area of true positives. This comes at the cost of a slightly delayed detection.


---

### Discussion

The main problem of the current approach is that performance is wasted by to many scales for the window search. These parameters should be fine-tuned more carefully on a bigger set of test images. A better placement of the windows along with a good choice of the scales should bring a significant performance improvement. In general, the whole approach has too many meta-parameters which makes finding an optimal implementation quite tedious, so a critical examination of the feature set might also be a good idea.
