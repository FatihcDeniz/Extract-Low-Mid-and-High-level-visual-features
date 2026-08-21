"""
This script extracts object presence information from images using the MaskFormer model for panoptic segmentation.

Reference: Cheng, B., Schwing, A., & Kirillov, A. (2021). Per-pixel classification is not all you need for semantic segmentation. Advances in neural information processing systems, 34, 17864-17875.
Model Page: https://huggingface.co/docs/transformers/en/model_doc/maskformer

"""
import torch 
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
from PIL import Image
import os 
import pandas as pd
from utils.utils import is_image, check_create_directory

# Use GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained MaskFormer processor and model from Hugging Face
processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-ade") 
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-ade").to(device) 

def panoptic_segmentation(image: Image) -> dict:
    """
    Perform panoptic segmentation on a given image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image in RGB format.

    Returns
    -------
    dict
        Dictionary containing:
        - 'segmentation': segmentation map
        - 'segments_info': metadata about detected segments.
    """
    # Process the image
    inputs = processor(images=image, return_tensors="pt")
    inputs = inputs.to(device)
    # Run the model
    with torch.no_grad():
        outputs = model(**inputs)
    # Convert model output to panoptic segmentation format
    predicted_panoptic_map = processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
  
    return predicted_panoptic_map

def object_presence(results):
    """
        Count occurrences of object categories in segmentation results.

        Parameters
        ----------
        results : dict
            Output from panoptic_segmentation function.

        Returns
        -------
        dict
            Dictionary mapping object label names to their counts.

    """
    
    # Extract label IDs from segmentation results
    segment_to_label = [segment['label_id'] for segment in results["segments_info"]]
    count_labels = {}
    
    # Count occurrences of each label
    for i in range(0, len(segment_to_label)):
        if model.config.id2label[segment_to_label[i]] not in count_labels:
            count_labels[model.config.id2label[segment_to_label[i]]] = 1
        else:
            count_labels[model.config.id2label[segment_to_label[i]]] += 1
    return count_labels

# Change this with the image location!!!!!!!!
image_loc = r"..\Images"

# Initialize data dictionary:
# - One column per label
# - One column for image names
data = {id_name:[] for id_key, id_name in model.config.id2label.items()}
data["image"] = []
# Check save location
save_loc = r"..\Extracted Visual Features\Mid"
check_create_directory(save_loc)
# Iterati through images
for directory in os.listdir(image_loc):
    loc = os.path.join(image_loc, directory)
    if os.path.isdir(loc):
        for image_loc_self in os.listdir(loc):
            image_loc = os.path.join(loc, image_loc_self)
            if is_image(image_loc):
                # Store image name
                data["image"].append(image_loc_self)

                # Load and preprocess image
                image = Image.open(image_loc).convert("RGB").resize((224, 224))

                # Perform segmentation
                results = panoptic_segmentation(image)

                # Extract object presence
                results = object_presence(results)

                # Create binary presence indicators for each label
                for label_name in data.keys():
                    if label_name != "image":
                        if label_name in results:
                            data[label_name].append(1)  # Object present
                        else:
                            data[label_name].append(0)  # Object absent

                print(f"Finished {image_loc}.")

            else:
                print(f"Can not open {image_loc}. It might be corrupted image. Check for it!!!")

data = pd.DataFrame.from_dict(data)
data.to_csv(os.path.join(save_loc, "objects.csv"), index = False)
        
                