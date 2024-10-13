/*************** Test on server, local is below. ***************/
/*
using UnityEngine;
using UnityEngine.Windows.WebCam;
using System.Collections;
using System.Threading.Tasks;
using System.Net.WebSockets;
using System.Threading;
using System;
using Microsoft.MixedReality.Toolkit.Input;  // !
using Microsoft.MixedReality.Toolkit.Utilities;  // !
using System.Text;
using MixedReality.Toolkit;

public class HoloLensCamera : MonoBehaviour
{
    private PhotoCapture photoCaptureObject = null;
    private Texture2D targetTexture;
    private ClientWebSocket webSocket;

    // Asynchronous initialization at start
    async void Start()
    {
        StartCamera(); // Start the camera
        await ConnectWebSocket(); // Asynchronous WebSocket connection
    }

    // Initialize the camera
    void StartCamera()
    {
        Resolution cameraResolution = PhotoCapture.SupportedResolutions.GetEnumerator().Current;
        targetTexture = new Texture2D(cameraResolution.width, cameraResolution.height);

        PhotoCapture.CreateAsync(false, delegate (PhotoCapture captureObject)
        {
            photoCaptureObject = captureObject;

            CameraParameters cameraParameters = new CameraParameters();
            cameraParameters.hologramOpacity = 0.0f;
            cameraParameters.cameraResolutionWidth = cameraResolution.width;
            cameraParameters.cameraResolutionHeight = cameraResolution.height;
            cameraParameters.pixelFormat = CapturePixelFormat.BGRA32;

            captureObject.StartPhotoModeAsync(cameraParameters, delegate (PhotoCapture.PhotoCaptureResult result) {
                Debug.Log("Camera ready");
                TakePhoto();  // Capture the first photo
            });
        });
    }

    // Capture and process the photo
    void TakePhoto()
    {
        photoCaptureObject.TakePhotoAsync(OnCapturedPhotoToMemory);
    }

    // Process the captured image and send it to the WebSocket server
    async void OnCapturedPhotoToMemory(PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame)
    {
        if (result.success)
        {
            photoCaptureFrame.UploadImageDataToTexture(targetTexture);
            byte[] imageData = targetTexture.EncodeToJPG();

            // Get right index finger tip coordinates
            Vector3 fingertipPosition = GetRightIndexFingerTipPosition();

            // Package image data and fingertip coordinates
            byte[] packagedData = PackageData(imageData, fingertipPosition);

            // Call network transmission functionality
            await SendDataOverWebSocket(packagedData);

            // Capture the next photo
            TakePhoto();
        }
        else
        {
            Debug.LogError("Photo capture failed");
        }
    }

    // Get right index finger tip coordinates
    Vector3 GetRightIndexFingerTipPosition()
    {
        MixedRealityPose pose;
        if (HandJointUtils.TryGetJointPose(TrackedHandJoint.IndexTip, Handedness.Right, out pose))
        {
            return pose.Position;
        }
        else
        {
            Debug.Log("Unable to detect right index finger tip");
            return Vector3.zero;
        }
    }

    // Package image data and fingertip coordinates
    byte[] PackageData(byte[] imageData, Vector3 fingertipPosition)
    {
        // Convert fingertip coordinates to string
        string positionString = $"{fingertipPosition.x},{fingertipPosition.y},{fingertipPosition.z}";

        // Convert the coordinate string to a byte array
        byte[] positionData = Encoding.UTF8.GetBytes(positionString);

        // Package length information (image data length and coordinate data length)
        int imageDataLength = imageData.Length;
        int positionDataLength = positionData.Length;

        byte[] imageDataLengthBytes = BitConverter.GetBytes(imageDataLength);
        byte[] positionDataLengthBytes = BitConverter.GetBytes(positionDataLength);

        // Combine all data
        byte[] packagedData = new byte[4 + 4 + imageDataLength + positionDataLength];
        Buffer.BlockCopy(imageDataLengthBytes, 0, packagedData, 0, 4);
        Buffer.BlockCopy(positionDataLengthBytes, 0, packagedData, 4, 4);
        Buffer.BlockCopy(imageData, 0, packagedData, 8, imageDataLength);
        Buffer.BlockCopy(positionData, 0, packagedData, 8 + imageDataLength, positionDataLength);

        return packagedData;
    }

    // Clean up resources on application quit
    async void OnApplicationQuit()
    {
        photoCaptureObject.StopPhotoModeAsync(OnStoppedPhotoMode);
        if (webSocket != null && webSocket.State == WebSocketState.Open)
        {
            await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, string.Empty, CancellationToken.None);
            webSocket.Dispose();
            webSocket = null;
        }
    }

    void OnStoppedPhotoMode(PhotoCapture.PhotoCaptureResult result)
    {
        photoCaptureObject.Dispose();
        photoCaptureObject = null;
    }

    // Establish WebSocket connection
    private async Task ConnectWebSocket()
    {
        webSocket = new ClientWebSocket();
        Uri serverUri = new Uri("ws://localhost:8765");

        await webSocket.ConnectAsync(serverUri, CancellationToken.None);
        Debug.Log("WebSocket connected");
    }

    // Send packaged data over WebSocket
    public async Task SendDataOverWebSocket(byte[] data)
    {
        if (webSocket == null || webSocket.State != WebSocketState.Open)
        {
            Debug.Log("WebSocket disconnected, reconnecting...");
            await ConnectWebSocket();  // Reconnect if not connected
        }

        if (webSocket.State == WebSocketState.Open)
        {
            await webSocket.SendAsync(new ArraySegment<byte>(data), WebSocketMessageType.Binary, true, CancellationToken.None);
            Debug.Log("Data sent to PC.");
        }
    }
}

*/




/*************** Test on local ***************/

using UnityEngine;
using UnityEngine.Windows.WebCam;
using System.Collections;
using System.Threading.Tasks;
using System.Net.WebSockets;
using System.Threading;
using System;

public class HoloLensCamera : MonoBehaviour
{
    private PhotoCapture photoCaptureObject = null;
    private Texture2D targetTexture;
    private ClientWebSocket webSocket;

    // Asynchronous initialization at start
    async void Start()
    {
        StartCamera(); // Start the camera
        await ConnectWebSocket(); // Asynchronous WebSocket connection
    }

    // Initialize the camera
    void StartCamera()
    {
        Resolution cameraResolution = PhotoCapture.SupportedResolutions.GetEnumerator().Current;
        targetTexture = new Texture2D(cameraResolution.width, cameraResolution.height);

        PhotoCapture.CreateAsync(false, delegate (PhotoCapture captureObject)
        {
            photoCaptureObject = captureObject;

            CameraParameters cameraParameters = new CameraParameters();
            cameraParameters.hologramOpacity = 0.0f;
            cameraParameters.cameraResolutionWidth = cameraResolution.width;
            cameraParameters.cameraResolutionHeight = cameraResolution.height;
            cameraParameters.pixelFormat = CapturePixelFormat.BGRA32;

            captureObject.StartPhotoModeAsync(cameraParameters, delegate (PhotoCapture.PhotoCaptureResult result) {
                Debug.Log("Camera ready");
                TakePhoto();  // Capture the first photo
            });
        });
    }

    // Capture and process the photo
    void TakePhoto()
    {
        photoCaptureObject.TakePhotoAsync(OnCapturedPhotoToMemory);
    }

    // Process the captured image and send it to the WebSocket server
    async void OnCapturedPhotoToMemory(PhotoCapture.PhotoCaptureResult result, PhotoCaptureFrame photoCaptureFrame)
    {
        if (result.success)
        {
            photoCaptureFrame.UploadImageDataToTexture(targetTexture);
            byte[] imageData = targetTexture.EncodeToJPG();

            // Call the network transmission function
            await SendImageOverWebSocket(imageData);

            // Capture the next photo
            TakePhoto();
        }
        else
        {
            Debug.LogError("Photo capture failed");
        }
    }

    // Clean up resources on application quit
    async void OnApplicationQuit()
    {
        photoCaptureObject.StopPhotoModeAsync(OnStoppedPhotoMode);
        if (webSocket != null && webSocket.State == WebSocketState.Open)
        {
            await webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, string.Empty, CancellationToken.None);
            webSocket.Dispose();
            webSocket = null;
        }
    }

    void OnStoppedPhotoMode(PhotoCapture.PhotoCaptureResult result)
    {
        photoCaptureObject.Dispose();
        photoCaptureObject = null;
    }

    // Establish WebSocket connection
    private async Task ConnectWebSocket()
    {
        webSocket = new ClientWebSocket();
        Uri serverUri = new Uri("ws://localhost:8765");

        await webSocket.ConnectAsync(serverUri, CancellationToken.None);
        Debug.Log("WebSocket connected");
    }

    // Send image data over WebSocket
    public async Task SendImageOverWebSocket(byte[] imageData)
    {
        if (webSocket == null || webSocket.State != WebSocketState.Open)
        {
            Debug.Log("WebSocket disconnected, reconnecting...");
            await ConnectWebSocket();  // Reconnect if not connected
        }

        if (webSocket.State == WebSocketState.Open)
        {
            await webSocket.SendAsync(new ArraySegment<byte>(imageData), WebSocketMessageType.Binary, true, CancellationToken.None);
            Debug.Log("Image data sent to PC.");
        }
    }
}
