import os
import re
import sys
import cv2
import numpy as np
import torch
import pandas as pd
from huggingface_hub import login
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)

os.environ["HUGGINGFACE_API_TOKEN"] = "<enter the token here>"


hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
login(token=hf_token)

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

device = torch.device("cuda")

# load the model
model_id = "google/paligemma-3b-mix-448"
model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, token=hf_token
)
processor = PaliGemmaProcessor.from_pretrained(model_id)


def extract_locations(data):
    
    return [
        "".join(filter(lambda x: x.startswith("<loc") and x.endswith(">"), s.split()))
        for s in data
    ]

def get_detections(paligemma_response, width, height):

    detections = paligemma_response.split(" ; ")
    cleaned_data = extract_locations(detections)
    parsed_coordinates = []

    for detection in cleaned_data:

        detection = detection.replace("<loc", "").split()
        coordinates = detection[0].split(">")
        coordinates = list(map(int, coordinates[:4]))

        coordinates = [int(coord) / 1024 for coord in coordinates]
        parsed_coordinates.append(coordinates)

    bboxes = []

    for coordinate in parsed_coordinates:
        y1, x1, y2, x2 = coordinate
        y1, x1, y2, x2 = map(round, (y1 * height, x1 * width, y2 * height, x2 * width))

        bboxes.append((x1, y1, x2, y2))
        
    return bboxes


def get_result(img, prompt):

    output_img = np.array(img)

    # Send text prompt and image as input.
    inputs = processor(
        text=prompt,
        images=img,
        padding="longest",
        do_convert_rgb=True,
        return_tensors="pt",
    ).to("cuda")

    model.to(device)
    inputs = inputs.to(dtype=model.dtype)

    # Get output
    with torch.no_grad():
        output = model.generate(**inputs, max_length=2048)

    paligemma_response = processor.decode(output[0], skip_special_tokens=True)[
        len(prompt) :
    ].lstrip("\n")
    

    if "detect" in prompt.lower() and "loc" in paligemma_response:
        
        label = prompt.split("detect")[-1]
        bboxes = get_detections(
            paligemma_response, output_img.shape[1], output_img.shape[0]
        )
        titles = [label] * len(bboxes)
        
    return {"paligemma_response": paligemma_response, "boxes":bboxes, "labels":titles}


def run_inference_image(path, prompt):

    opencv_img = cv2.cvtColor(
        cv2.imread(path), cv2.COLOR_BGR2RGB
    )
    
    img = Image.fromarray(opencv_img)
    response = get_result(img, prompt)

    return response


def create_via_csv_row(image_file, file_size, bbox, cls, region_id):
    """
    Creates a single row for the VIA CSV format.

    Parameters:
    - image_file (str): The name of the image file.
    - file_size (int): The size of the image file in bytes.
    - bbox (list): A list representing the bounding box [x_min, y_min, width, height].
    - cls (str): The class label for the object in the bounding box.
    - region_id (int): The ID of the region (bounding box) within the image.

    Returns:
    - dict: A dictionary representing a single annotation row in VIA CSV format.
    """
    x_min, y_min, width, height = bbox
    return {
        'filename': image_file,
        'file_size': file_size,
        'file_attributes': '',
        'region_count': '',
        'region_id': region_id,
        'region_shape_attributes': f'{{"name":"rect","x":{x_min},"y":{y_min},"width":{width},"height":{height}}}',
        'region_attributes': f'{{"class":"{cls}"}}'
    }


def generate_via_csv(file_paths, bboxes, classes, output_csv_file):
    """
    Generates a VIA-compatible CSV file from the given bounding box annotations.

    Parameters:
    - file_paths (list): A list of file paths to the images.
    - bboxes (list): A list of lists, where each sublist contains bounding box coordinates for an image.
    - classes (list): A list of lists, where each sublist contains class labels corresponding to the bounding boxes.
    - output_csv_file (str): The file path where the CSV output will be saved.

    Returns:
    - None: Saves the CSV file at the specified path.
    """
    annotations = []
    
    for i, file_path in enumerate(file_paths):
        file_size = os.path.getsize(file_path)  # Get file size
        
        # Loop through bounding boxes and classes
        for region_id, (bbox, cls) in enumerate(zip(bboxes[i], classes[i])):
            row = create_via_csv_row(file_path, file_size, bbox, cls, region_id)
            annotations.append(row)
    
    # Convert annotations to DataFrame and save to CSV
    df = pd.DataFrame(annotations)
    df.to_csv(output_csv_file, index=False)
    print(f"CSV Annotations saved to {output_csv_file}")


# Example usage of the script
if __name__ == "__main__":
    # Sample input data
    
    image_folder = "images"
    output_file = "via_annotations.csv" 
    prompt = "detect wind turbine"
    
    file_paths = os.listdir(image_folder)
    file_paths = [f"{image_folder}/{name}" for name in file_paths]
    
    bboxes = []
    classes = []
    
    for file_path in tqdm(file_paths):
        
        response = run_inference_image(file_path, prompt)
        bbox = response["boxes"]
        class_values = response["labels"]
        
        bboxes.append(bbox)
        classes.append(class_values)
        
    
    generate_via_csv(file_paths, bboxes, classes, output_file)
