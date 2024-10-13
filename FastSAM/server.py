import asyncio
import websockets
from data_parser import parse_received_data
from inference_HOLO import run_inference
import cv2
from flask import Flask, request, jsonify
import threading
from flask_cors import CORS  # Add CORS support

# Define WebSocket server address and port
SERVER_ADDRESS = '0.0.0.0'
SERVER_PORT = 8765

# Create Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Part status dictionary, False means not picked up, True means picked up
part_status = {
    'part_1': False,
    'part_2': False,
    'part_3': False,
    'part_4': False,
    'part_5': False,
    'part_6': False,
    'part_7': False,
    'part_8': False,
    'part_9': False,
    'part_10': False,
    'part_11': False,
    'part_12': False,
    'part_13': False,
    'part_14': False,
}
status_lock = threading.Lock()

# Robotic arm control function, assuming you have already implemented it
def pick_up_part(part_name):
    # Call your robotic arm control code here
    print(f"Robotic arm is starting the pick-up operation for {part_name}...")
    # TODO: Add robotic arm control logic

@app.route('/pick_up', methods=['POST'])
def pick_up():
    data = request.json
    part_name = data.get('part_name')
    if part_name is None:
        return jsonify({'status': 'failed', 'message': 'No part name provided'}), 400

    with status_lock:
        if part_status.get(part_name) == False:
            # Update part status
            part_status[part_name] = True

            # Perform the robotic arm pick-up operation (can be executed in a new thread to avoid blocking)
            threading.Thread(target=pick_up_part, args=(part_name,)).start()

            return jsonify({'status': 'success', 'part_name': part_name})
        else:
            return jsonify({'status': 'failed', 'message': 'Part already picked up'}), 400

@app.route('/get_status', methods=['GET'])
def get_status():
    with status_lock:
        status_list = [{'key': k, 'value': v} for k, v in part_status.items()]
        response = jsonify({'items': status_list})
        response.headers.add('Access-Control-Allow-Origin', '*')  # Allow cross-origin requests
        return response

async def handle_connection(websocket, path):
    print("HoloLens 2 is connected")
    try:
        while True:
            # Receive data sent from HoloLens 2
            data = await websocket.recv()
            print("Data received, parsing...")

            # Parse the data to get the initial image and fingertip coordinates
            image, initial_fingertip_position = parse_received_data(data)

            # Define callback function to capture new image
            def capture_new_image():
                # Put logic to capture new image here. Can capture from a camera or simulate.
                # Example: Use a new image file for testing
                new_image = cv2.imread('new_image_after_yolo.jpg')  # Replace with your logic to get a new image
                return new_image

            # Define callback function to capture new fingertip coordinates
            def capture_new_fingertip_position():
                # Put logic to capture new fingertip coordinates here, simulate or get from HoloLens
                new_fingertip_position = (150, 200)  # Example coordinates, replace with actual data from the device
                return new_fingertip_position

            # Run inference to get grid number and other info, pass the callback functions to capture new images and fingertip coordinates
            results = run_inference(image, initial_fingertip_position, capture_new_image, capture_new_fingertip_position)

            # Inference results contain whether a gesture was detected and which grid the fingertip is in
            if results['gesture_detected']:
                grid_number = results.get('grid_number', None)  # Get grid number from inference results
                
                if grid_number is not None:
                    # Determine which part to operate based on grid number
                    part_name = f"part_{grid_number}"  # Assuming grid numbers correspond to part numbers

                    # Confirm the part has not been picked up yet
                    with status_lock:
                        if part_status.get(part_name, False) is False:
                            # Perform the robotic arm pick-up operation
                            threading.Thread(target=pick_up_part, args=(part_name,)).start()
                            part_status[part_name] = True  # Update the status to picked up

                            # Actively push the status update to the client
                            await websocket.send(f"Part {part_name} has been picked up")

                    # Send grid number back to the client (HoloLens 2)
                    await websocket.send(f"Grid {grid_number} - {part_name} will be picked up")
                else:
                    await websocket.send("Unable to determine grid number")
            else:
                await websocket.send("Gesture type B not detected")

    except websockets.exceptions.ConnectionClosed:
        print(f"HoloLens 2 connection closed: {websocket.remote_address}")
    except Exception as e:
        print(f"An error occurred: {e}")

def start_websocket_server():
    asyncio.run(websockets.serve(handle_connection, SERVER_ADDRESS, SERVER_PORT))

def start_flask_app():
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    # Start WebSocket server
    websocket_thread = threading.Thread(target=start_websocket_server)
    websocket_thread.start()

    # Start Flask server
    flask_thread = threading.Thread(target=start_flask_app)
    flask_thread.start()
