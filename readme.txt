0. Install the DeTR required package: 
$ conda install -c pytorch pytorch torchvision pycocotools albumentations

(Dependencies for Keras U-Net 3+: TensorFlow 2.5.0, Keras 2.5.0, Numpy 1.19.5.)

1. Please download these two files from the google drive and extract them to the corresponding folder.

2. For object detection (DeTR model), please download the file from: https://drive.google.com/file/d/1sT3V0Zl_CmVaoYZ0T2ou6AsWT5VOC1DB/view?usp=share_link (12.08Gb)

3. For CRF and seg, please download from:
 i: https://drive.google.com/file/d/14HWw79KvTvkahCTtUlbcSSvOJK593cCW/view?usp=share_link (5.09Gb)
 ii: https://drive.google.com/file/d/1rHUUeEnQR8t0y64ZGl4rJxTAqXA1Mc1k/view?usp=share_link (1Gb)

4. For testing object detection algorithm, please remain "obj_detection" folder file structure unchanged and run the run-3D.py

5. For CRF, simply running the included ipynb should be okay

6. For seg (U-Net 3+), run the 3plus Unet.py. (But remember to delete the raw prediction) Also note our focus is on CRF and DeTR. Segmentation is just a preliminary model.

Note that  miscellaneous folder are only for illustration purpose. They are part of our project but not the focus of this course.
It is just for showing our effort to the course.

We suggest to test the code in Colab environment as we use pytorch for DeTR and Keras for U-Net 3+ (again just the preliminary)

Please focus on CRF and obj_detection folders only. They are the main progress of our project.
