**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Normalize the features and randomize a selection for training and testing.
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/example_data.jpg
[image2]: ./output_images/hog-visualization.jpg
[image3]: ./output_images/sliding-windows.jpg
[image4]: ./output_images/raw-detection.jpg
[image5]: ./output_images/heatmap.jpg
[image6]: ./output_images/labels.jpg
[image7]: ./output_images/final.jpg
[video1]: ./project_video.mp4

### Histogram of Oriented Gradients (HOG)

#### Extract HOG features from the training images.

In order to create a robust vehicle detection system, feature extraction was performed on the training data before fitting it to a classifier. The code for extracting hog features is contained in the fourth code cell of the IPython notebook in the function 'extract_features'.  

Here is an example of the output of `skimage.hog()` using the YCrCb color space and HOG parameters of `orientations=9`, `pixels_per_cell=8` and `cells_per_block=(2, 2)`:

![alt text][image2]

The training data is a combination of the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video.  The following is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]


Using the following parameters resulting in both the best accuracy of the trained model against the test set and the most accurate vehicle identifacation on the test images.

color_space = 'YCrCb'
orient = 9  
pix_per_cell = 8 
cell_per_block = 2 
hog_channel = "ALL" 
spatial_size = (20, 20) 
hist_bins = 16    
spatial_feat = True 
hist_feat = True 
hog_feat = True 
y_start_stop = [450, None] 
ystart = 400
ystop = 656
scale = 1.5

#### Classifier training.
To further improve the performance of vehicle detection, YCrCb color features and spatial features were extracted (cell four). The training data was split into random training and test sets using scikit learn's 'train_test_split' and then normalized to zero mean and equal variance.

A linear support vector machines classifier (SVM) was trained using these features(cell seven).  

### Sliding Window Search

The sliding window search is conducted in code cell 5. A window with a tunable scale moves across an image and each time the classifier runs.  The scale parameter was tweaked iteratively, and 1.8 was found to give the most accurate classifcations on test images.

![alt text][image3]

#### Inference pipeline and optimization

The inference pipeline involves first extracting features for the entire image using YCrCb 3-channel HOG features. The sliding window search is then performed on the HOG feature extracted image and 
The color and spatial features are also extracted in each window so that the SVM has the same preprocessing steps applied as the training phase. Each window is classified by the SVM as 'car' or 'not car'.
Here is an example classification.

![alt text][image4]

These bounding boxes are stored in memory. A heatmap was created of the last fifty:

![alt text][image5]
 
`scipy.ndimage.measurements.label()` wasthen  used to label the areas of highest heat
![alt text][image6]

Finally, a single box is drawn over each label
![alt text][image7]
---

### Video Implementation

Here is a [link to the video result](./test_videos_output/project_video.mp4)

As was described in the pipeline, the positions of positive detections were stored for each frame of the video. The last fifty were used to create a heatmap and then apply a threshold to that map to identify vehicle positions. 

### Discussion

YCrCb was chosen because it contains a luminance channel in addition to chroma channels. Previous computer vision attempts with gradient thresholding have demonstrated that different lighting might be challenging for classifiers. This alone made a big difference in the performance of the detection.

A weakness of the pipeline is that the inference stage is slower than desirable, limiting it's real world usability. This may be improved by extracting color and spatial features before performing the sliding window search. The pipeline relies on detected objects repeating in frames to reject false positives, so if a 'not car' object occurs in many consecuative frames it may be prone to false positives. 

To make the pipeline more robust, more sophistocated machine learning algorithms or even deep learning may be used instead of SVM. 


