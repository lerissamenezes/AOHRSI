# Reproduced Projects of Peers

1. Semantic Segmentation of Roofs from Aerial Imagery using Deep Learning
Imagery used is from Kaggle and the work is done in Google Colab. I have tried to reproduce the results for segmentation using different batch_sizes and epochs for training the data.
Training Set 1: Batch Size 5, Epochs 10
The predictions after 10 epochs are a clear improvement over the initial state. The model successfully identified the general areas of the roofs in the test images. However, the predicted masks are noticeably blurry and lack the crisp, sharp edges seen in the ground truth images. The model appears to be outputting probabilities for each pixel rather than confident binary values.
 <img width="940" height="400" alt="image" src="https://github.com/user-attachments/assets/2dc1b3b8-56ac-40e7-be2d-c5c864d0f7fb" />


Training Set 2: Batch Size 12, Epochs 15
The predictions after 15 epochs show a slight improvement in the clarity of the segmented roofs compared to the 10-epoch run. The model has learned to define the edges a little more sharply. However, the predictions still show some blurriness and don't perfectly match the ground truth.
 <img width="940" height="380" alt="image" src="https://github.com/user-attachments/assets/e0fdf2bd-38ba-4215-83f2-adf7f83a213c" />

https://colab.research.google.com/drive/1pqmlQDj4WbUtP25AmeOWHbCX7zDVLlVp?usp=sharing
Training Set 3: Batch Size 14, Epochs 30
After 30 epochs, the predictions still exhibit the same blurriness and lack of fine detail. The model's performance on the validation set, according to the provided training log, reached a val_accuracy of 0.9301 and val_loss of 0.0026 by Epoch 1, and these values remain stable through to Epoch 40. This indicates that additional epochs, even up to 30 or 40, do not significantly improve the model's ability to generalize to new data.
 <img width="940" height="344" alt="image" src="https://github.com/user-attachments/assets/702e2cfa-2079-49bd-a381-1d60667ded19" />


Training Set 3: Batch Size 16, Epochs 40
This set was used by Ayesha for the final submission, but I found that the model performs optimally when the epoch is set to 15, after which the predictions turned to be grainy. More epoch didn’t give me a better prediction for the rooftops segmentation.
 <img width="940" height="373" alt="image" src="https://github.com/user-attachments/assets/567d3b76-a745-4452-873c-1587dcd10d4a" />

Although the segmentation model showed high accuracy for each training set, the predicted masks are blurry. This can be due to the resizing of images to 128 x 128 pixels, which may have left out some rooftop details. Majority of the ground truth has black background with only small portion occupying the rooftops i.e whote portion. This may have enabled the model to pick up the black portion extensively and leave out crucial details from the rooftops. 

2. Damage Detection of Buildings caused by the Blatten Glacier
They have used a Laplacian filter for the initial processing of raw images. This is because the pre- and post-images may look the same on the raw bands, but damaged houses usually create sharp local changes in reflectance or texture. Applying a Laplacian filter emphasizes those sudden local differences, such as roof collapse edges, debris outlines, and cracks. This makes segmentation more sensitive to structural changes. The filter has a kernel size 3 on Band 1(Blue) and Band 3(Red). 
Band 1 is sensitive to atmospheric scattering and the Laplacian on Blue enhances small structural changes. Band 3 is sensitive to vegetation and soil reflectance and the Laplacian on Red enhances contrast between damaged surfaces and surroundings.
Following this, the GLCM (Gray Level Co-occurrence Matrix) filter is applied on band 1. This is particularly helpful in enhancing textural changes from smooth intact roofs to rough broken roofs.
After this, the imagery is clipped to the central area of interest where the majority of the damage has occurred. All four clipped images are then stacked to form a 6-band raster image for each case, pre- and post-event. The final stacked image is fed to the MeanShift Segmentation algorithm in the OTB. Here, spatial radius that defines. Spatial radius and Range radius is 15 and 14, respectively, which means that the model is more tolerant towards grouping objects together. The radius chosen is not too small or too big so that it preserves the details of pre- and post-damage events. This step produces the polygons.
The next step is where the zonal statistics i.e mean, standard deviation, minimum and maximum are calculated. These parameters are used to calculate the compactness. Since houses and roads have similar spectral properties, segmentation can’t clearly separate compact houses from long, thin road segments. Hence, compactness is calculated only on the pre-event image. Compactness was set to 1, which roughly represents a circle, so that only houses are extracted from the pre-event imagery and these were compared with that of post-event. Low compactness represents the roads, as they are thin and elongated.
The sample labels are then joined with the segmented polygons to produce a training dataset i.e a class label is attached to statistical attributes of the segment. This dataset is then used to train the SVM and samples for pre- and post-event are trained separately. The model is designed in QGIS in the Model Designer.
The model gives different classes as output and only the houses classes is extracted from the pre-event imagery and glacier and lake classes from the post imagery. These extracted classes are then intersected to evaluate the damaged houses.
 <img width="810" height="645" alt="image" src="https://github.com/user-attachments/assets/162d0b4a-09da-45b2-a45d-b69874ae0f46" />

The orange area represents the damaged houses, submerged by the lake and glacier. The pink area represents the intact houses.
The pink area represents the intersection between the Pre-event and Post-event houses only. The orange area is the difference of the Pre-event and the intersection.
By visual inspection, the model classifies doesn’t leave out any houses, but there are a few false positives. It takes into consideration the flora around the houses.
These workflows have been automated to work within a single model and give the desired output. 
 
3. Road Segmentation using Machine Learning
I have tested the accuracy of the model created by Darian for all the cities. But the Deepness model kept crashing on my system. I converted the tile resolution, yet deepness showing the error. OverflowError: cannot convert float infinity to integer.
The tile size used is 0.2m x 0.2m. 
After processing the imagery, the thick black line represents the clipped highways. The entire image is broken down into 81 tiles each and then subjected to the evaluation of the model, where we get the F1 score and IoU Index. This is because the entire image is too large for processing. 
The ortho tif image is used to produce a predicted mask. This predicted mask along with Ortho and OSM tif images is used to generate a binary image as well as the 81 tiles. These tiles are used to validate the model and obtain the F1 score and IoU index. 
 <img width="596" height="562" alt="image" src="https://github.com/user-attachments/assets/9eebcf67-9af5-46b2-8697-9b4f6093f584" />

This is the imagery for Muenster, after clipping.
IoU and F1 score that I have obtained is 0.5001 and 0.6539 in contrast with the 0.5862 and 0.7290 obtained by Darian, using the same model but I have obtained these values for Warendorf. Lower F1 score indicates that the model is either missing roads (false negatives) or predicting extra roads (false positives).
IoU represents Intersection over Union, which means that there is 50% overlap of the predicted mask over the ground truth mask. F1 score combines the precision and recall and there is 65% similarity between the ground truth and predicted mask for Muenster.
Muenster:
 <img width="531" height="338" alt="image" src="https://github.com/user-attachments/assets/62171d51-5602-49d9-a4f0-57caa2a31bb7" />

Considering the IoU and F1 score as provided by Darian for the Deepness model, the own model gives better results than the Deepness.
Ahlen:
 <img width="591" height="404" alt="image" src="https://github.com/user-attachments/assets/3133a6b9-ac3b-4ef5-81ea-cbaa53e46a2b" />

Dorsten:
 <img width="649" height="380" alt="image" src="https://github.com/user-attachments/assets/0c0af3d8-71ea-4c00-be98-2e1621273d31" />

Duelmen: 
 <img width="598" height="381" alt="image" src="https://github.com/user-attachments/assets/bfa769bf-eb94-4bae-8f43-0a9218aabb68" />

Hamm:
 <img width="557" height="359" alt="image" src="https://github.com/user-attachments/assets/b154bb39-1e74-472f-8f10-65b95f910e85" />

Warendorf:
 <img width="816" height="434" alt="image" src="https://github.com/user-attachments/assets/d4e68114-5d05-49c7-b145-c690c1a01d3d" />

 
4. Road Segmentation using Computer Vision
 <img width="940" height="324" alt="image" src="https://github.com/user-attachments/assets/2f948496-82ac-41b3-b5e9-605765c2bf1b" />

The method used by Ayemen for road segmentation is not yielding the desired results, as the final output is grainy and doesn’t segment roads. The script provided doesn’t include any steps to generate the ground truth, and calculate the accuracy metrics. The expected outcome is to produce the road segments but the script creates a mask over the background.

