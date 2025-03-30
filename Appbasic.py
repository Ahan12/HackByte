from ultralytics import YOLO
import cv2
import numpy as np
import threading
from datetime import datetime
import csv
from PIL import Image
from kivy.app import App
from kivy.uix.camera import Camera
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.clock import Clock, mainthread
import torch
from torchvision import transforms
from torchvision.ops import nms
from collections import deque

def letterbox_image(image, target_size=(640, 640), fill_color=(128, 128, 128)):
    """Resize image with unchanged aspect ratio using padding."""
    iw, ih = image.size
    w, h = target_size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image_resized = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', target_size, fill_color)
    left = (w - nw) // 2
    top = (h - nh) // 2
    new_image.paste(image_resized, (left, top))
    return new_image

def process_yolo_output(raw_output, conf_threshold=0.3, iou_threshold=0.45):
    """
    Processes raw YOLO output of shape [1, 84, 8400] into a list of detections.
    Each detection is a tuple: (x1, y1, x2, y2, confidence, class_idx).
    """
    raw_output = raw_output.permute(0, 2, 1)
    raw_output = raw_output[0]  # shape: [8400, 84]

    boxes = raw_output[:, :4]
    obj_logits = raw_output[:, 4]
    class_logits = raw_output[:, 5:]

    objectness = torch.sigmoid(obj_logits)
    class_scores = torch.sigmoid(class_logits)

    max_class_scores, class_indices = torch.max(class_scores, dim=1)
    overall_scores = objectness * max_class_scores

    print(f"Max overall detection score: {overall_scores.max().item():.3f}")

    mask = overall_scores > conf_threshold
    if not mask.any():
        return []

    filtered_boxes = boxes[mask]
    filtered_scores = overall_scores[mask]
    filtered_class_indices = class_indices[mask]

    cx = filtered_boxes[:, 0]
    cy = filtered_boxes[:, 1]
    w  = filtered_boxes[:, 2]
    h  = filtered_boxes[:, 3]
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    bboxes = torch.stack([x1, y1, x2, y2], dim=1)

    keep = nms(bboxes, filtered_scores, iou_threshold)
    bboxes = bboxes[keep]
    final_scores = filtered_scores[keep]
    final_class_indices = filtered_class_indices[keep]

    detections = []
    for i in range(len(bboxes)):
        x1_, y1_, x2_, y2_ = bboxes[i].tolist()
        score_ = final_scores[i].item()
        cls_idx = int(final_class_indices[i].item())
        detections.append((x1_, y1_, x2_, y2_, score_, cls_idx))
    return detections

class YOLOCombinedApp(App):
    def build(self):
        layout = BoxLayout(orientation="vertical", padding=10, spacing=10)
        self.title_label = Label(text="Combined YOLO Detection & Classification", font_size="24sp", size_hint=(1, 0.1))
        layout.add_widget(self.title_label)
        
        self.camera = Camera(play=True, resolution=(640, 480), size_hint=(1, 0.6))
        layout.add_widget(self.camera)
        
        self.result_label = Label(text="Waiting for classification...", font_size="20sp", size_hint=(1, 0.1))
        layout.add_widget(self.result_label)
        
        self.log_label = Label(text="", font_size="16sp", size_hint=(1, 0.2))
        layout.add_widget(self.log_label)
        
        self.det_model = YOLO("yolov8n.pt")  # Default YOLOv8 model
        
        self.custom_model = torch.jit.load("C:\\Users\\samne\\OneDrive\\ADVAY\\Food Waste\\Phone_App\\Custom.pt")
        self.custom_model.eval()
        
        self.custom_transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ])
        
        self.det_transform = transforms.Compose([
            transforms.Lambda(lambda img: letterbox_image(img, target_size=(640, 640))),
            transforms.ToTensor(),
        ])
        
        # Complete list of 80 COCO classes
        self.coco_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
            5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
            10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
            14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
            20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
            25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
            30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
            35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
            39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
            44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
            49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
            54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
            59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
            64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
            69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
            74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
            79: "toothbrush"
        }

        # Custom model's class names (all are food items)
        self.custom_names = {
            0: "apple_pie", 1: "baby_back_ribs", 2: "baklava", 3: "beef_carpaccio", 4: "beef_tartare",
            5: "beet_salad", 6: "beignets", 7: "bibimbap", 8: "bread_pudding", 9: "breakfast_burrito",
            10: "bruschetta", 11: "caesar_salad", 12: "cannoli", 13: "caprese_salad", 14: "carrot_cake",
            15: "cheesecake", 16: "chicken_curry", 17: "chicken_wings", 18: "chocolate_cake", 19: "chocolate_mousse",
            20: "churros", 21: "clam_chowder", 22: "club_sandwich", 23: "crab_cakes", 24: "creme_brulee",
            25: "croque_madame", 26: "cup_cakes", 27: "deviled_eggs", 28: "donuts", 29: "dumplings",
            30: "edamame", 31: "eggs_benedict", 32: "falafel", 33: "filet_mignon", 34: "fish_and_chips",
            35: "foie_gras", 36: "french_fries", 37: "french_onion_soup", 38: "fried_calamari", 39: "fried_rice",
            40: "frozen_yogurt", 41: "garlic_bread", 42: "gnocchi", 43: "greek_salad", 44: "grilled_cheese_sandwich",
            45: "grilled_salmon", 46: "guacamole", 47: "gyoza", 48: "hamburger", 49: "hot_and_sour_soup",
            50: "hot_dog", 51: "huevos_rancheros", 52: "hummus", 53: "ice_cream", 54: "lasagna",
            55: "lobster_bisque", 56: "lobster_roll_sandwich", 57: "macaroni_and_cheese", 58: "macarons", 59: "miso_soup",
            60: "mussels", 61: "nachos", 62: "omelette", 63: "onion_rings", 64: "paella",
            65: "pancakes", 66: "panna_cotta", 67: "peking_duck", 68: "pho", 69: "pizza",
            70: "pork_chop", 71: "poutine", 72: "prime_rib", 73: "pulled_pork_sandwich", 74: "ramen",
            75: "ravioli", 76: "red_velvet_cake", 77: "risotto", 78: "samosa", 79: "sashimi",
            80: "scallops", 81: "seaweed_salad", 82: "shrimp_and_grits", 83: "spaghetti_bolognese", 84: "spaghetti_carbonara",
            85: "spring_rolls", 86: "steak", 87: "stir_fry", 88: "sushi", 89: "tacos",
            90: "takoyaki", 91: "tiramisu", 92: "tuna_tartare", 93: "waffles"
        }
        
        self.log_filename = "combined_detection_log.csv"
        self.init_log()
        
        # For temporal smoothing, keep a history of detections (last 5 frames)
        self.detections_history = deque(maxlen=5)
        
        # Schedule frame capture every 0.2 seconds.
        Clock.schedule_interval(self.capture_frame, 0.2)
        return layout

    def init_log(self):
        try:
            with open(self.log_filename, mode="x", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Timestamp", "Source", "Detection", "Confidence"])
        except FileExistsError:
            pass

    def capture_frame(self, dt):
        if self.camera.texture:
            try:
                texture = self.camera.texture
                size = texture.size
                data = texture.pixels
                image = Image.frombytes("RGBA", size, data).convert("RGB")
                threading.Thread(target=self.process_frame, args=(image.copy(),), daemon=True).start()
            except Exception as e:
                self.update_result(f"Frame capture error: {e}")

    def process_frame(self, image):
        try:
            det_tensor = self.det_transform(image).unsqueeze(0)
            custom_tensor = self.custom_transform(image).unsqueeze(0)
            
            # Run YOLO Detection
            img_np = np.array(image)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            det_results = self.det_model(img_cv, stream=True)
            detection_text = "No detection"
            det_conf = 0.0
            yolo_label = ""
            for r in det_results:
                boxes = r.boxes
                if len(boxes) > 0:
                    best_box = max(boxes, key=lambda b: b.conf.item())
                    det_conf = best_box.conf.item()
                    cls = int(best_box.cls.item())
                    yolo_label = self.det_model.names[cls]
                    detection_text = f"{yolo_label} ({det_conf:.2f})"
                    break

            # Run Custom Classification Model
            with torch.no_grad():
                custom_out = self.custom_model(custom_tensor)
                if isinstance(custom_out, (list, tuple)):
                    custom_out = custom_out[0]
                probs = torch.softmax(custom_out, dim=1)
                custom_conf, custom_idx = torch.max(probs, dim=1)
                custom_conf = custom_conf.item()
                custom_class = self.custom_names.get(int(custom_idx.item()), "Unknown")
                custom_text = f"{custom_class} ({custom_conf:.2f})"
            
            print(f"Detection result: {detection_text}")
            print(f"Custom classification result: {custom_text}")

            # Determine the final result.
            # Create allowed food set by taking the union of the custom model's classes and
            # the default YOLO (COCO) food classes.
            allowed_food_coco = {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"}
            allowed_food = set(self.custom_names.values()).union(allowed_food_coco)
            custom_label = custom_text.split(" ")[0]
            
            if yolo_label in allowed_food:
                final_source = "Detection"
                final_text = detection_text
                final_conf = det_conf
            elif custom_label in allowed_food:
                final_source = "Custom"
                final_text = custom_text
                final_conf = custom_conf
            else:
                final_source = None
                final_text = "No food detected"
                final_conf = 0.0

            # Update UI and log (require average confidence >= 30%)
            if final_source is not None:
                self.detections_history.append((final_source, final_text, final_conf))
                votes = {}
                for src, txt, conf in self.detections_history:
                    label = txt.split(" ")[0]
                    votes[label] = votes.get(label, 0) + 1
                if votes:
                    final_label = max(votes, key=votes.get)
                    confs = [conf for src, txt, conf in self.detections_history if txt.split(" ")[0] == final_label]
                    final_conf_avg = sum(confs) / len(confs)
                    final_text = f"{final_label} ({final_conf_avg:.2f})"
                else:
                    final_conf_avg = final_conf

                if final_conf_avg < 0.30:
                    print("No food item detected with high enough confidence, not logging.")
                    self.update_result("No food detected")
                else:
                    print("Final result:", final_text)
                    self.update_result(final_text)
                    self.log_result(final_source, final_text, final_conf_avg)
            else:
                print("No food detected, not logging.")
                self.update_result(final_text)
        except Exception as ex:
            self.update_result(f"Inference error: {ex}")
            print("Error:", ex)

    @mainthread
    def update_result(self, text):
        self.result_label.text = text

    def log_result(self, source, detection, confidence):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_filename, mode="a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([timestamp, source, detection, confidence])
        except Exception as e:
            print(f"Logging error: {e}")

if __name__ == "__main__":
    YOLOCombinedApp().run()
