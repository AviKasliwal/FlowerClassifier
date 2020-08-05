import streamlit as st
from PIL import Image
import pickle
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.applications import VGG16
from keras import models
from keras import layers
from keras import optimizers

st.title ("🌼 Flower Classifier")
st.subheader('A Deep Learning based image recognition appplication to classify a flower into one of the 17 classes.')

st.header ("🖥️ Demo !")

def load_model ():
	vgg_conv = VGG16(weights = 'imagenet', include_top = False, input_shape = (224, 224, 3))
	for layer in vgg_conv.layers[: - 4]:
		layer.trainable = False
	model = models.Sequential()
	model.add (vgg_conv)
	model.add(layers.Flatten())
	model.add(layers.Dense(1024, activation = 'relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(512, activation = 'relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(17, activation = 'softmax'))
	model.load_weights ("ModelWeights/weights2.hdf5")
	return model

uploaded_file = st.file_uploader ("Choose an image...", type = "jpg")

def get_key(val): 
	for key, value in flowers_dict.items(): 
		if val == value: 
			return key 

flowers_dict = {'BlueShell': 0, 'Buttercip': 1, 'ColtsFoot': 2, 'Cowslip': 3, 'Crocus': 4, 'Daffodil': 5, 'Daisy': 6, 'Dandelion': 7, 'Fritillary': 8, 'LilyValley': 9, 'Pansy': 10, 'Snowdrop': 11, 'Sunflower': 12, 'TigerLily': 13, 'Tulip': 14, 'WindFlower': 15, 'iris': 16}

if uploaded_file is not None:
	uploaded = Image.open (uploaded_file)
	st.image (uploaded, caption = "Uploaded image", use_column_width = True)
	image = np.array (uploaded.resize ((224, 224)))
	image = np.expand_dims (image, axis = 0)
	image = preprocess_input (image)
	model = load_model ()
	pred = model.predict (image)
	pred = list (pred[0])
	label = get_key (pred.index (1))
	st.success (f"Predicted label for the image is : {label}")
	


st.header ("🔖 Introduction")
st.write ("Classifying flowers is a difficult task even for the humans – surely more difficult than distnguishing a boy from a tree from an airplane.")

image = Image.open ("Images/intro1.png")
st.image (image, caption = "Image source : A Visual Vocabulary for Flower Classification - Maria-Elena Nilsback and Andrew Zisserman", use_column_width = True)

st.markdown ("For example in the image above, to a human the **left** most flower and the **middle** flower seem to be of **same** type and the **right** most flower of some other type, but the **left** and **right** images are both **Dandelions**. The **middle** one is a **Colts’ foot**.")

st.subheader ("What makes flower classification a challenging task?")

st.write ("* Flowers images have huge variations in :   \n",
	"   * View Point  \n",
	"   * Scale   \n",
	"   * Illumination  \n",
	"   * Partial occlusions  \n",
	"   * Multiple instances  \n",
	"   * Cluttered backgrounds etc...  \n")
	
st.write ("* While classifying flowers there is a chance that we classify the background content rather than the flower itself.  \n")

st.write ("* But the greatest challenge occurs when the inter-class variability is less than the intra-class variability. As seen from the example of Dandelions & Colts  \n")

st.subheader ("How do the Botanists classify flowers?")
st.write ("Botanists use keys, where a series of questions need to be answered in order to classify a flower. In most cases some of the question are related to internal structure that can only by made visible by disecting the flower.") 

st.write ("But for a visual object classification problem this is not possible. It is possible however to narrow down the choices to a short list of plausible flowers.")

st.subheader ("What distinguishes one flower from another can sometimes be their :")

st.write ("  * Shape \n",
	"  * Color \n",
	"  * Distinctive texture patterns \n",
	"  * Or a combination of these 3 aspects. \n")

st.write ("The challenge lies in finding a good representation for these aspects and a way of combining them that preserves the distinctiveness of each aspect, rather than averaging over them")

st.write ("And to extract the above mentioned features from a an image of a flower I'll be using a CNN based architecture (transfer learning) attached to a linear classifier for classifying the flower as one of the 17 flowers.")

st.header ("🔖 Dataset")
st.write ("The dataset, consists of 17 flower categories. The flowers in the datasaet are the flowers that are commonly occuring in the United Kingdom. Each class consists of 80 images.")

st.write ("From this data I created my own datset by keeping 65 images of each class for training and the remaining 15 for validation.")

st.write ("The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories")

st.write ("Link for the dataset : http://www.robots.ox.ac.uk/~vgg/data/flowers/17/")

image = Image.open ("Images/data.png")
st.image (image, caption = "Data Structure", use_column_width = True)

image = Image.open ("Images/train17.png")
st.image (image, caption = "Train Images Sample", use_column_width = True)

image = Image.open ("Images/valid17.png")
st.image (image, caption = "Validation Images Sample", use_column_width = True)

st.header ("🔖 Model")
st.write ("Since my dataset was small but the images were similar to the images of imagenet dataset, therefore I decided to use transfer learning for classifying my images.")

st.subheader ("Model Architecture")
image = Image.open ("Images/modelArchitecture.png")
st.image (image, caption = "Model Architecture", use_column_width = True)
st.write ("Model is a sequential model created using Keras, consits of two parts : ")
st.write ("* Convolution Base  \n",
"   * Input = (batch x 224 x 224 x 3)  \n",
"   * The convolution base acts as a feature extractor where the top layers extract the low level features and the bottom layers extract high level features as they have higer reception field.  \n",
"   * Output = (batch x 7 x 7 x 512)  \n")

st.write ("* Classifier  \n",
"   * Input = The output of convbase is flatten and then fed into the classifier , shape (batch x 25088) \n",
"   * The linear classifier consisted of 2 hidden layers followed by dropouts to reduce overfitting.  \n",
"   * Output = (batch x 17)  \n")

st.subheader ("Model Summary")
image = Image.open ("Images/modelSummary.png")
st.image (image, caption = "Model Summary", use_column_width = True)

st.markdown ("> All but last 4 layers of the convilution base were **non trainable**")

st.markdown ("> The Learning rate was kept **low** so that the learnt weights were not destroyed / changed significantly.")

st.markdown ("> **ImageDataGenerator** was used to feed the data into the model with **image augmentaion** on the fly to make the model more robust.")

st.subheader ("🔖 Results")

st.markdown ("Train accuracy = **88.73%**   \n Test accuracy = **82.74%**")

st.markdown ("> These results were obtained after 50 epochs.")

st.markdown ("> Best model weights were saved using **ModelCheckpoint** callback")

st.markdown ("___")

st.subheader ("🔖 Further Tasks")
st.write ("  * Use LRScheduler \n",
	"  * Reduce the depth of MLP \n",
	"  * Improve the results for images where the background is dominant than the flower. \n",
	"  * Try other model architectures \n",
	"  * Try with more Data. \n",
	"  * Deploy to the web \n")
st.markdown ("___")	
