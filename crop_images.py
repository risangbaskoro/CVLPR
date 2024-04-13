#!/usr/bin/env python

import xml.etree.ElementTree as et

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def get_bounding_boxes(xml_path):
    tree = et.parse(xml_path)
    root = tree.getroot()

    bounding_boxes = []
    for obj in root.findall('object'):
        text = obj.find('name').text.split('-')[-1].upper()
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        bounding_boxes.append((text, (xmin, ymin, xmax, ymax)))

    return bounding_boxes


def crop_images(source:str = "Datasets/cvlpr",
                img_size: tuple = (94, 24), 
                destination: str = "data"):
    # Path definition
    path = Path(source)
    imgs_path = path / "images"
    xmls_path = path / "annotations"

    # Make a directory called images, skip if it already exists
    dest_path = Path.cwd() / destination
    dest_path.mkdir(exist_ok=True)

    # List of all image names
    img_names = [img.name for img in imgs_path.iterdir()]

    num_images_without_annotation = 0
    num_plates_without_label = 0
    
    for img_name in tqdm(img_names):
        xml_path = xmls_path / f"{img_name.split('.')[0]}.xml"

        if not xml_path.exists():
            num_images_without_annotation += 1
        else:
            bounding_boxes = get_bounding_boxes(xml_path)

            # Read image using PIL
            img = Image.open(imgs_path / img_name)

            # Loop over bounding boxes
            for bbox in bounding_boxes:
                name, (x_min, y_min, x_max, y_max) = bbox
                # Crop the portion of the image inside the bounding box
                cropped_image = img.crop((x_min, y_min, x_max, y_max))
                cropped_image = cropped_image.resize(img_size)
                # Save the cropped image
                i = 0
                while (dest_path / f"{name}_{i}.jpg").exists():
                    i += 1

                if name == "PLATE":
                    num_plates_without_label += 1
                    continue
                cropped_image.save(dest_path / f"{name}_{i}.jpg")

    if num_images_without_annotation:
        print(f"WARNING: \t {num_images_without_annotation} images does not have annotations.")
    if num_plates_without_label:
        print(f"WARNING: \t {num_plates_without_label} plates does not have label.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-s", "--source", type=str, default=Path.home()/"Datasets/dataset", help="Path to the dataset")
    parser.add_argument("-a", "--size", type=tuple, default=(94, 24), help="Size of the cropped images")
    parser.add_argument("-d", "--destination", type=str, default="cropped", help="Path to save the cropped images")
    args = parser.parse_args()

    crop_images(args.source, args.size, args.destination)
