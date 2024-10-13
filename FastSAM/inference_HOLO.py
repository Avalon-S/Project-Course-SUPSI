import cv2
from ultralytics import YOLO
from fastsam import FastSAM, FastSAMPrompt
import torch
from PIL import Image
from grid_detector import GridDetector
import os
import time  # Import the time module for delays

# Initialize YOLO and FastSAM models
yolo_model = YOLO('./_YOLOV8_HandGesture/best.pt') 
fastsam_model = FastSAM('./weights/FastSAM-x.pt')  

# Define default parameters for FastSAM
class Args:
    imgsz = 1024
    conf = 0.4
    iou = 0.9
    text_prompt = "white tray holding black mechanical parts"
    box_prompt = [[0, 0, 0, 0]]
    point_prompt = [[0, 0]]
    point_label = [0]
    retina = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    withContours = False
    better_quality = False
    output_folder = './output/'
    randomcolor = True

args = Args()

def run_inference(image, initial_fingertip_position, capture_new_image_callback, capture_new_fingertip_position_callback):
    """
    Run YOLOv8 and FastSAM inference on the received image, and use GridDetector to determine 
    which grid the fingertip coordinates belong to.
    
    Parameters:
    - image: OpenCV image object
    - initial_fingertip_position: Initial fingertip position as an (x, y) tuple
    - capture_new_image_callback: Callback function to capture a new image
    - capture_new_fingertip_position_callback: Callback function to capture new fingertip coordinates

    Returns:
    - results: A dictionary containing the inference results, including the grid number.
    """
    # Run YOLOv8 inference
    yolo_results = yolo_model.predict(source=image, save=False, conf=0.5)

    # Process YOLOv8 results
    gesture_detected = False
    yolo_boxes = []
    for result in yolo_results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id == 1:  # Assuming class ID 1 is for gesture type B
                gesture_detected = True
                yolo_boxes.append(box.xyxy[0].tolist())  # Convert tensor to list

    if gesture_detected:
        # If gesture type B is detected, capture a new image after 2 seconds
        time.sleep(2)
        image = capture_new_image_callback()  # Capture new image for FastSAM

        # Run FastSAM inference
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image = Image.fromarray(input_image)

        # FastSAM model inference
        everything_results = fastsam_model(
            input_image,
            device=args.device,
            retina_masks=args.retina,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou
        )

        # Initialize prompt processing
        prompt_process = FastSAMPrompt(input_image, everything_results, device=args.device)

        # Apply FastSAM prompt
        if args.box_prompt[0][2] != 0 and args.box_prompt[0][3] != 0:
            ann = prompt_process.box_prompt(bboxes=args.box_prompt)
        elif args.text_prompt is not None:
            ann = prompt_process.text_prompt(text=args.text_prompt)
        elif args.point_prompt[0] != [0, 0]:
            ann = prompt_process.point_prompt(
                points=args.point_prompt, pointlabel=args.point_label
            )
        else:
            ann = prompt_process.everything_prompt()

        # Get the output image path from FastSAM
        output_image_path = os.path.join(args.output_folder, "fastsam_output.jpg")
        prompt_process.plot(
            annotations=ann,
            output_path=output_image_path,
            bboxes=args.box_prompt,
            points=args.point_prompt,
            point_label=args.point_label,
            withContours=args.withContours,
            better_quality=args.better_quality
        )

        # Load the generated segmented image
        segmented_image = cv2.imread(output_image_path)

        # Capture new fingertip position after 2 seconds
        time.sleep(2)
        fingertip_pos = capture_new_fingertip_position_callback()  # Capture new fingertip position
        fingertip_pos = (int(fingertip_pos[0]), int(fingertip_pos[1]))

        # Use GridDetector to determine which grid the fingertip belongs to
        grid_detector = GridDetector()  # Instantiate the GridDetector class
        grid_number = grid_detector.detect_grid(segmented_image, fingertip_pos)  # Call detect_grid with the segmented image and new fingertip coordinates

        # Return inference results
        results = {
            'gesture_detected': True,
            'yolo_boxes': yolo_boxes,
            'fingertip_position': fingertip_pos,
            'grid_number': grid_number  # Return the grid number
        }
    else:
        results = {
            'gesture_detected': False,
            'message': 'No gesture type B detected'
        }

    return results
