import os
import argparse
from fastsam import FastSAM, FastSAMPrompt 
import ast
import torch
from PIL import Image
from utils.tools import convert_box_xywh_to_xyxy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM-x.pt", help="model"
    )
    parser.add_argument(
        "--input_folder", type=str, default="./images/", help="path to input image folder"
    )
    parser.add_argument(
        "--output_folder", type=str, default="./output/", help="folder to save results"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()

def process_image(image_path, model, args):
    # Load image
    input = Image.open(image_path)
    input = input.convert("RGB")
    
    # Run model
    everything_results = model(
        input,
        device=args.device,
        retina_masks=args.retina,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou    
    )
    
    # Initialize prompt processing
    prompt_process = FastSAMPrompt(input, everything_results, device=args.device)
    
    # Apply prompts
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

    # Save output
    output_path = os.path.join(args.output_folder, os.path.basename(image_path))
    prompt_process.plot(
        annotations=ann,
        output_path=output_path,
        bboxes=args.box_prompt,
        points=args.point_prompt,
        point_label=args.point_label,
        withContours=args.withContours,
        better_quality=args.better_quality,
    )
    print(f"Processed and saved: {output_path}")

def main(args):
    # Create output folder if it doesn't exist
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # Load model
    model = FastSAM(args.model_path)
    
    # Parse point and box prompts
    args.point_prompt = ast.literal_eval(args.point_prompt)
    args.box_prompt = convert_box_xywh_to_xyxy(ast.literal_eval(args.box_prompt))
    args.point_label = ast.literal_eval(args.point_label)
    
    # Process each image in the input folder
    for file_name in os.listdir(args.input_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(args.input_folder, file_name)
            process_image(image_path, model, args)

if __name__ == "__main__":
    args = parse_args()
    main(args)

