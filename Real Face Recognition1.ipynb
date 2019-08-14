{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Face Recognition using OpenCV."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This script details the process for building a face recognition system by bounding box method using OpenCV."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "#This changes your working directory to the folder that has your training-data and test-data folders\n",
    "#You can change this to the folder of your choice\n",
    "os.chdir('Desktop/DATA SCIENCE MASTERS/PROJ/')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['s1', 's2']"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#List the directories and make sure your training-data folder has 2 folders inside with names s1 and s2 respectively\n",
    "os.listdir('training-data')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['1.jpg',\n",
       " '10.jpg',\n",
       " '2.jpg',\n",
       " '3.jpg',\n",
       " '4.jpg',\n",
       " '5.jpg',\n",
       " '6.jpg',\n",
       " '7.jpg',\n",
       " '8.jpg',\n",
       " '9.jpg']"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Check the s1 folder to make sure it has at least 10 images labelled as so 1.jpg, 2.jpg, etc\n",
    "os.listdir('training-data/s1')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['1.jpg',\n",
       " '10.jpg',\n",
       " '2.jpg',\n",
       " '3.jpg',\n",
       " '4.jpg',\n",
       " '5.jpg',\n",
       " '6.jpg',\n",
       " '7.jpg',\n",
       " '8.jpeg',\n",
       " '9.jpeg']"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Check the s2 folder to make sure it also has at least 10 images labelled as so 1.jpg, 2.jpg, etc\n",
    "os.listdir('training-data/s2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['test1.jpg', 'test2.jpg']"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "os.listdir('test-data')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Lets import some modules\n",
    "import cv2\n",
    "import os # for reading training data directories and paths\n",
    "import numpy as np # to convert python lists to numpy arrays as it is needed by OpenCV face recognizers\n",
    "\n",
    "\n",
    "#Prepare training data\n",
    "#this should be in folders labelled with the names of the people to train the model with\n",
    "#The more images used in the training, the better\n",
    "#I'll use at least 10 images in each folder\n",
    "#However the test folder should just contain images with no labels\n",
    "\n",
    "#there is no label 0 in our training data, so subject name for index/label 0 is empty\n",
    "subjects = [\"\", \"Vangelis Michael\", \"Jennifer Rex\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Now lets prepare the data\n",
    "#Function to detect faces using openCV\n",
    "def detect_face(img):\n",
    "    #Convert the images to gray scale as openCV expects gray images\n",
    "    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
    "    \n",
    "    \n",
    "#load OpenCV face detector, i'm using HAAR which is slower but works best in this notebook in detecting faces\n",
    "#you could also use another accurate and faster classifier: LBP\n",
    "#    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')\n",
    "    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') #for haar\n",
    "\n",
    "#Lets detect multiscale images\n",
    "#result should be a list of faces\n",
    "    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5);\n",
    "\n",
    "#if no faces are detected then return original img\n",
    "    if (len(faces) == 0):\n",
    "        return None, None\n",
    "\n",
    "#under the assumption that there will be only one face,\n",
    "#extract the face area\n",
    "    (x, y, w, h) = faces[0]\n",
    "\n",
    "#return only the face part of the image\n",
    "    return gray[y:y+w, x:x+h], faces[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "#MAIN WORK\n",
    "def prepare_training_data(data_folder_path):\n",
    "    #STEP 1\n",
    "    #Get the directories(one directory for each subject) in data folder\n",
    "    dirs = os.listdir(data_folder_path)\n",
    "    \n",
    "    #list to hold all subject faces and labels\n",
    "    faces = []\n",
    "    labels =[]\n",
    "    \n",
    "    #Lets go through each directory and read images within it\n",
    "    for dir_name in dirs:\n",
    "        #our subject directories start with letter 's' so ignore any non-relevant directories if any\n",
    "        if not dir_name.startswith(\"s\"):\n",
    "            continue;\n",
    "        #STEP 2\n",
    "        #extract label number of subject from dir_name\n",
    "        #format of dir name = slabel\n",
    "        #so removing letter 's' from dir_name will give us label           \n",
    "        label = int(dir_name.replace(\"s\", \"\"))\n",
    "        \n",
    "        #build path of directory containing images for current subject \n",
    "        #sample subject_dir_path = 'traning-data/s1'\n",
    "        subject_dir_path = data_folder_path + \"/\" + dir_name\n",
    "        \n",
    "        #Get the image names that are inside the iven subject directory\n",
    "        subject_image_names = os.listdir(subject_dir_path)\n",
    "        \n",
    "        #STEP 3\n",
    "        #go through each image name, read image, detect face and add face to list of faces\n",
    "        for image_name in subject_image_names:\n",
    "            #ignore system files like .DS_Store\n",
    "            if image_name.startswith(\".\"):\n",
    "                continue;\n",
    "            \n",
    "            #build image path\n",
    "            #sample image path = training-data/s1/1.pgm\n",
    "            image_path = subject_dir_path + \"/\" + image_name\n",
    "            #read image\n",
    "            image = cv2.imread(image_path)\n",
    "            \n",
    "            #Display an image window to show the image\n",
    "            cv2.imshow(\"Training on image...\", cv2.resize(image, (400, 500)))\n",
    "            cv2.waitKey(100)\n",
    "            #detect face\n",
    "            face, rect = detect_face(image)\n",
    "            \n",
    "            #STEP 4\n",
    "            #For the purpose of this tutorial, we will ignore faces that are not detected\n",
    "            if face is not None:\n",
    "                #     add face to list of faces\n",
    "                faces.append(face)\n",
    "                #add label for this face\n",
    "                labels.append(label)\n",
    "                \n",
    "    cv2.destroyAllWindows()\n",
    "    cv2.waitKey(1)\n",
    "    cv2.destroyAllWindows()\n",
    "    \n",
    "    return faces, labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Preparing data...\n",
      "Data prepared\n",
      "Total faces:  13\n",
      "Total labels:  13\n"
     ]
    }
   ],
   "source": [
    "#Lets prepare our training data\n",
    "#data will be on two lists of the same size\n",
    "#One list will contain all the faces\n",
    "#and the other list will contain respective labels for each face\n",
    "print('Preparing data...')\n",
    "faces, labels = prepare_training_data(\"training-data\")\n",
    "print('Data prepared')\n",
    "\n",
    "#Print total faces and labels\n",
    "print('Total faces: ', len(faces))\n",
    "print(\"Total labels: \", len(labels))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Lets Train our face recognizer.\n",
    "#Below are the 3 face recognizers openCV comes equiped with\n",
    "#EigenFaces: cv2.face.createEigenFaceRecognizer()\n",
    "#FisherFaces: cv2.face.createFisherFaceRecognizer()\n",
    "#Local Binary Patterns Histogram (LBPH): cv2.face.LBPHFisherFaceRecognizer()\n",
    "\n",
    "#i'll use the LBPH recognizer\n",
    "\n",
    "#create our LBPH recognizer\n",
    "face_recognizer = cv2.face.LBPHFaceRecognizer_create()\n",
    "#face_recognizer = cv2.face.EigenFaceRecognizer_create() #All train images must be of equal size dimensions\n",
    "#face_recognizer = cv2.face.FisherFaceRecognizer_create() # All train images must also be of equal size pixel 1461681 pixels\n",
    "\n",
    "\n",
    "#Train our face recognizer of our training faces\n",
    "face_recognizer.train(faces, np.array(labels))\n",
    "#OpenCV expects labels vector to be a numpy array"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Predictions\n",
    "#Function to draw rectangle on image\n",
    "#according to given (x, y) coordinates and \n",
    "#given width and height\n",
    "def draw_rectangle(img, rect):\n",
    "    (x, y, w, h) = rect\n",
    "    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)\n",
    "    \n",
    "#function to draw text on given image starting from passed (x, y) coordinates.\n",
    "def draw_text(img, text, x, y):\n",
    "    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)\n",
    "    \n",
    "\n",
    "#This function recognizes the person in image passed\n",
    "#and draws a rectangle around detected face with name\n",
    "#of the subject\n",
    "def predict(test_img):\n",
    "    \n",
    "#make a copy of the image as we dont want to change original image\n",
    "    img = test_img.copy()\n",
    "    \n",
    "#detect face from the image\n",
    "    face, rect = detect_face(img)    \n",
    "\n",
    "#predict the image using our face recognizer\n",
    "    label, confidence = face_recognizer.predict(face)\n",
    "#get name of respective label returned by face recognizer\n",
    "    label_text = subjects[label]\n",
    "\n",
    "#draw a rectangle around face detected\n",
    "    draw_rectangle(img, rect)\n",
    "#draw name of predicted person\n",
    "    draw_text(img, label_text, rect[0], rect[1]-5)\n",
    "    \n",
    "    return img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicting images...\n",
      "Prediction Complete\n"
     ]
    }
   ],
   "source": [
    "print(\"Predicting images...\")\n",
    "\n",
    "#load test images\n",
    "test_img1 = cv2.imread('test-data/test1.jpg')\n",
    "test_img2 = cv2.imread('test-data/test2.jpg')\n",
    "\n",
    "#perform a prediction\n",
    "predicted_img1 = predict(test_img1)\n",
    "predicted_img2 = predict(test_img2)\n",
    "print('Prediction Complete')\n",
    "\n",
    "#display both images\n",
    "#cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))\n",
    "cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))\n",
    "cv2.waitKey(0)\n",
    "cv2.destroyAllWindows()\n",
    "cv2.waitKey(1)\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:Python3]",
   "language": "python",
   "name": "conda-env-Python3-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
