import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import torch

#from torch.cuda import device
from ultralytics import YOLO


# Select GPU if available, else fallback to CPU
default_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 OCR will run on: {default_device}")

class KSAPlateExtractor:
    def __init__(self, model_path, conf=0.05, device=default_device):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

        #print("ocr_model:", self.model.names)

    def get_detections_from_array(self, img_array):
        results = self.model.predict(img_array, conf=self.conf, device=self.device)[0]
        boxes = getattr(results, 'obb', None) or getattr(results, 'boxes', None)
        if boxes is None or boxes.cls is None or boxes.cls.numel() == 0:
            print(f"❌ No detections found for plate crop.")
            return []
        class_ids = boxes.cls.cpu().numpy().astype(int)
        coords = boxes.xyxy.cpu().numpy()
        names = [self.model.names[c] for c in class_ids]
        detections = []
        for cid, name, coord in zip(class_ids, names, coords):
            detections.append({'name': name, 'bbox': coord})
        return detections

    def extract_plate_left_to_right(self, detections):
        detections_sorted = sorted(detections, key=lambda d: (d['bbox'][0] + d['bbox'][2]) / 2)
        area_code = []
        license_number = []
        digit_classes = {f'En_{i}' for i in range(10)}
        for det in detections_sorted:
            name = det['name']
            if name in digit_classes:
                license_number.append(name.replace('En_', ''))
            elif name.startswith('En_'):
                area_code.append(name.replace('En_', '').upper())
        plate_number = ''.join(area_code) + ' ' + ''.join(license_number)
        return plate_number.strip(), area_code, license_number, detections_sorted

    def process_plate_array(self, crops, visualize=True):
        detections = self.get_detections_from_array(crops)
        if not detections:
            return None
        plate_number, area_code, license_number, detections_sorted = self.extract_plate_left_to_right(detections)
        print("***************************************************")
        print(f"Area Code        : {''.join(area_code)}")
        print(f"License Number   : {''.join(license_number)}")
        print("***************************************************")
        return {
            "area_code": ''.join(area_code),
            "license_number": ''.join(license_number),
            "plate_number": f"{''.join(license_number)} {''.join(area_code)}"
        }
        
    def detect_and_recognize_plates(self, car_image, plate_model, visualize=False):
        """
        Unified method to detect license plates in a car image and perform OCR on them.
        
        Args:
            car_image: Input car image (numpy array)
            plate_model: YOLO model for license plate detection
            visualize: Whether to visualize the results
            
        Returns:
            List of dictionaries containing OCR results for each detected plate
        """
        # Detect license plates in the car image
        crops = self.get_plate_crops_from_car_image(car_image, plate_model)
        
        results = []
        for idx, crop_img in enumerate(crops):
            print(f"\nProcessing Plate Crop #{idx + 1}")
            # Perform OCR on each detected plate
            ocr_result = self.process_plate_array(crop_img, visualize=visualize)
            if ocr_result:
                results.append(ocr_result)
                
        return results


    def get_plate_crops_from_car_image(self,orig_img, plate_model):
        results = plate_model.predict(orig_img, conf=0.2, device=self.device, task='obb')[0]

        boxes = results.obb
        if not hasattr(boxes, 'cls') or boxes.cls is None:
            return []
        classes = boxes.cls.cpu().numpy().astype(int)
        xywhr = boxes.xywhr.cpu().numpy()
        plate_class_idx = None
        for k, v in plate_model.names.items():
            if v == "LP":
                plate_class_idx = k
                break
        if plate_class_idx is None:
            print("No 'plate' class found in model!")
            return []
        crops = []
        for i in range(len(classes)):
            if int(classes[i]) == plate_class_idx:
                x, y, w, h, a = xywhr[i]
                angle_deg = math.degrees(float(a)) if abs(a) < 3.2 else float(a)
                rect = ((x, y), (w, h), angle_deg)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                x_min, y_min = np.min(box, axis=0)
                x_max, y_max = np.max(box, axis=0)
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(orig_img.shape[1], x_max), min(orig_img.shape[0], y_max)
                crop = orig_img[int(y_min):int(y_max), int(x_min):int(x_max)]
                if crop.size > 0 and crop.shape[0] > 5 and crop.shape[1] > 5:
                    crops.append(crop)
        return crops

"""
if __name__ == "__main__":
    # 1. Load models
    plate_model = YOLO("/kaggle/input/lp_detection_plate/pytorch/default/1/lp_ksa_plate_detection.pt")
    ocr_model = KSAPlateExtractor(
        model_path="/kaggle/input/ksa_ocr/pytorch/default/1/lp_ksa_ocr.pt",
        conf=0.05,
        device="cpu"
    )
    print("plate detection classes:", plate_model.names)

    car_images_dir = "/kaggle/input/saudi-test/test/images"  # Update as needed!

    # 2. Process each car image
    for fname in os.listdir(car_images_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(car_images_dir, fname)
            orig_img = cv2.imread(img_path)
            crops = get_plate_crops_from_car_image(orig_img, plate_model)
            for idx, crop_img in enumerate(crops):
                print(f"\nImage: {fname} - Plate Crop #{idx + 1}")
                ocr_result = ocr_model.process_plate_array(crop_img, visualize=True)
                # ocr_result is a dict with area_code, license_number, plate_number
"""