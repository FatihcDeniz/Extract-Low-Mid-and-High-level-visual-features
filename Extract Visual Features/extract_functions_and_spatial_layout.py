
"""
This script extracts high-level semantic features (scene functions and spatial layout)
from images using the Places365 WideResNet model. Some of the code were taken from
https://github.com/csailvision/places365, so please also cite them.

"""


import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms 
from torch.nn import functional as F
import os
import numpy as np
from PIL import Image
import pandas as pd
from utils.utils import is_image, check_create_directory

os.environ['KMP_DUPLICATE_LIB_OK']='True'


# hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def load_labels() -> tuple:
    """
    Load scene categories, indoor/outdoor labels, and scene attributes.
    Returns
    -------
    tuple
        (classes, labels_IO, labels_attribute, W_attribute)
    """
    # Load labels for the model
    file_name_category = r'.\utils\functions and spatial layout\categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = r'.\utils\functions and spatial layout\IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = r'.\utils\functions and spatial layout\labels_sunattribute.txt'
    if not os.access(file_name_attribute, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
        os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = r'.\utils\functions and spatial layout\W_sceneattribute_wideresnet18.npy'
    if not os.access(file_name_W, os.W_OK):
        synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
        os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute

def hook_feature(module, input, output):
    """
    Forward hook to extract intermediate CNN features.

    Parameters
    ----------
    module : torch.nn.Module
    input : tuple
    output : torch.Tensor

    Notes
    -----
    """
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnTF():
    """
    Create image preprocessing pipeline.

    Returns
    -------
    torchvision.transforms.Compose
        Transformation pipeline for Places365 model.
    """

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transform


def load_model():
    """
    Load pretrained Places365 WideResNet model.
    Returns
    -------
    torch.nn.Module
        Model ready for inference with feature hooks.
    """
    model_file = r'.\utils\functions and spatial layout\wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    import utils.wideresnet
    model = utils.wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
    
    model.eval()

    model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model

def load_sunattributes():
    """
    Load SUN attribute labels used for function extraction.

    Returns
    -------
    list
        List of attribute labels.
    """

    with open(r".\utils\functions and spatial layout\labels_sunattribute.txt","r") as sun_functions:
        sun_functions = sun_functions.readlines()
        sun_functions = [i.strip() for i in sun_functions]
    return sun_functions

# Load the sun attributes
sun_functions = load_sunattributes()
classes, labels_IO, labels_attribute, W_attribute = load_labels()
img_scores = {}
count = 0
# Load model
model = load_model()
transform = returnTF() 
params = list(model.parameters())
weight_softmax = params[-2].data.numpy()
weight_softmax[weight_softmax<0] = 0
print("LOADED MODEL")

# Change this with the image location!!!!!!!!
image_loc = r"..\Images"

# Check save location
save_loc_functions = r"..\Extracted Visual Features\High"
save_loc_layout = r"..\Extracted Visual Features\Mid"
check_create_directory(save_loc_functions)
check_create_directory(save_loc_layout)

# Iterate through images
for folder in os.listdir(image_loc):
    image_folder  = os.path.join(image_loc, folder)
    for image in os.listdir(image_folder):
        if is_image(os.path.join(image_folder, image)):
            features_blobs = []
            # Load and process images
            img = Image.open(os.path.join(image_folder, image))
            img = V(transform(img).unsqueeze(0))
            # Forward pass
            logit = model.forward(img)
            # Softmax probabilities
            h_x = F.softmax(logit,1 ).data.squeeze()
            # calculate probabilities for each scene category. 
            probs, idx = h_x.sort(0, True)
            probs = probs.numpy()
            idx = idx.numpy()
            label_dict = {}            
            # Get model outputs for functions by calculating dot product of 
            # attribute vector and output from the last layer
            responses_attribute = W_attribute.dot(features_blobs[1])

            # Apply sigmoid function to function outputs
            responses_attribute = F.sigmoid(torch.from_numpy(responses_attribute)).data.squeeze()
            responses_attribute = responses_attribute.numpy()
            idx_a = np.argsort(responses_attribute)
            for i in range(-1,-len(responses_attribute)-1,-1):
                if labels_attribute[idx_a[i]] in sun_functions:
                    label_dict[labels_attribute[idx_a[i]]] = responses_attribute[idx_a[i]]
                    
            img_scores[image] = label_dict
            print(f"Finished {image}.")
                    
image_scores = pd.DataFrame(img_scores).T
image_scores = image_scores.reset_index()
image_scores.rename(columns = {"index":"image"}, inplace = True)

# List of functions extracted from Places
functions = ["boating","driving","biking","transporting","sunbathing","touring","hiking","climbing","camping",
             "reading","studying","training","research","diving","swimming","bathing","eating","cleaning",
             "socializing","congregating","waiting in line","competing","sports","exercise","playing","gaming",
             "spectating","farming","constructing","shopping","medical activity","working","using tools","digging",
             "conducting business","praying", "image"]
# List of spatial layout features extracted from Places
layout = ["open area", "semi-enclosed area", "enclosed area", "far-away horizon", "no horizon", "image"]

functions_data = image_scores.loc[:, image_scores.columns.isin(functions)]
layout_data = image_scores.loc[:, image_scores.columns.isin(layout)]

functions_data.to_csv(os.path.join(save_loc_functions, "functions.csv"), index = False)
layout_data.to_csv(os.path.join(save_loc_layout, "layout.csv"), index = False)






