# Face-detection-model
This is a face detection model that I'll try to improve testing different models and approches

# specifications
all tests are done on lenovo ideapad310 with i5-7500U and WITHOUT using the GPU

# generate refreence embeddings
put the refreence images in "ref images" sorted in the same order of the refrence dictionary
example: "ref images/0.jpg" is the first name in the refrence dictionay

# version 1
using SSD ResNet100 and FaceNet built on Inception V1, avg FPS~7

# version 2
using YOLO and FaceNet built on Inception V1, avg FPS~11.

# known bug for now: 
~~code crash when detect multi faces in the same frame~~
to generate ref embeddings you need to put the images both in the ref folder AND one directory up it (right next to the model files)

# version 3
used face tracking algorithm instead of running face recognition all the time which gave a really big boost in performancec the code now achieves 27~29 fps on RP3 and 45 on i5-7500U without charger
used insightface project bufflo_sl based on mobilefacenet for both detection and trained with ArcFace for recognition
fixed the ref embeddings code, now you need to put the images in ref_images folder and name them with each individual name ex (mustafa.jpg) and run the code