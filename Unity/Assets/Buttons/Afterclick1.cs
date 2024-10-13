/*************** Test on server, local is below. ***************/
/*
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using System.Collections;
using System.Text;
using System.Collections.Generic;

public class AfterClick : MonoBehaviour
{
    // Array of button objects
    public Button[] buttons;

    // Set the disabled color (color will change permanently after click)
    public Color disabledColor = Color.gray;

    // Array of Image components for the buttons
    private Image[] buttonImages;

    // List of part names, corresponding to the buttons
    private string[] partNames = {
        "part_1", "part_2", "part_3", "part_4", "part_5", "part_6", "part_7",
        "part_8", "part_9", "part_10", "part_11", "part_12", "part_13", "part_14"
    };

    // Server URL
    private string serverUrl = "http://192.168.x.x:5000"; // Server IP address

    // Array to prevent duplicate clicks
    private bool[] isButtonClicked;

    // Robotic arm controller object
    public RoboticArmController roboticArm;

    void Start()
    {
        buttonImages = new Image[buttons.Length];
        isButtonClicked = new bool[buttons.Length];  // Initialize the array to prevent duplicate clicks

        for (int i = 0; i < buttons.Length; i++)
        {
            int index = i;
            buttonImages[i] = buttons[i].GetComponent<Image>();

            if (buttons[i] != null)
            {
                buttons[i].onClick.AddListener(() => OnButtonClick(index));
            }
        }

        // Get the part status at startup
        StartCoroutine(GetPartStatus());

        // Set a timer to periodically get part status and update the buttons
        InvokeRepeating("RequestPartStatusUpdate", 0.1f, 0.1f);  // Request status update every 0.1 seconds
    }

    // Function triggered when a button is clicked
    public void OnButtonClick(int buttonIndex)
    {
        // If the button has already been clicked, don't process it again
        if (isButtonClicked[buttonIndex])
        {
            Debug.Log("Button " + (buttonIndex + 1) + " has already been clicked.");
            return;
        }

        string partName = partNames[buttonIndex];
        Debug.Log("Button " + (buttonIndex + 1) + " is clicked to start picking up " + partName + "...");

        // Send the pick-up request
        StartCoroutine(PickUpPart(partName, buttonIndex));

        // Invoke the robotic arm control logic
        if (roboticArm != null)
        {
            roboticArm.PickUpItem(buttonIndex);
        }
    }

    IEnumerator PickUpPart(string partName, int buttonIndex)
    {
        string url = serverUrl + "/pick_up";
        var jsonData = "{\"part_name\":\"" + partName + "\"}";
        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);

        UnityWebRequest request = new UnityWebRequest(url, "POST");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            Debug.Log("Successfully requested to pick up " + partName);
            // Disable the button to prevent duplicate clicks
            DisableButton(buttonIndex);
            // Mark the button as clicked
            isButtonClicked[buttonIndex] = true;
        }
        else
        {
            Debug.Log("Error in picking up " + partName + ": " + request.error);
        }
    }

    // Periodically get part status
    void RequestPartStatusUpdate()
    {
        StartCoroutine(GetPartStatus());
    }

    IEnumerator GetPartStatus()
    {
        string url = serverUrl + "/get_status";
        UnityWebRequest request = UnityWebRequest.Get(url);
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonResult = request.downloadHandler.text;
            Debug.Log("Part status: " + jsonResult);

            // Parse the JSON and update button status
            Dictionary<string, bool> statusDict = ParsePartStatus(jsonResult);
            UpdateButtons(statusDict);
        }
        else
        {
            Debug.Log("Error in getting part status: " + request.error);
        }
    }

    Dictionary<string, bool> ParsePartStatus(string json)
    {
        PartStatusWrapper wrapper = JsonUtility.FromJson<PartStatusWrapper>(json);
        return wrapper.ToDictionary();
    }

    void UpdateButtons(Dictionary<string, bool> statusDict)
    {
        for (int i = 0; i < partNames.Length; i++)
        {
            string partName = partNames[i];
            if (statusDict.ContainsKey(partName) && statusDict[partName])
            {
                // The part has already been picked up, disable the button
                DisableButton(i);
            }
        }
    }

    void DisableButton(int buttonIndex)
    {
        buttons[buttonIndex].interactable = false;

        // Set the button color to the disabled color permanently
        if (buttonImages[buttonIndex] != null)
        {
            buttonImages[buttonIndex].color = disabledColor;
        }

        // Record the status to prevent the button from being clicked again
        isButtonClicked[buttonIndex] = true;
    }
}

// Robotic arm control class, assuming this is the logic for controlling the robotic arm
public class RoboticArmController : MonoBehaviour
{
    // Perform different operations based on the button index
    public void PickUpItem(int buttonIndex)
    {
        switch (buttonIndex)
        {
            case 0:
                Debug.Log("Robotic arm starts picking up part 1...");
                break;
            case 1:
                Debug.Log("Robotic arm starts picking up part 2...");
                break;
            case 2:
                Debug.Log("Robotic arm starts picking up part 3...");
                break;
            case 3:
                Debug.Log("Robotic arm starts picking up part 4...");
                break;
            case 4:
                Debug.Log("Robotic arm starts picking up part 5...");
                break;
            case 5:
                Debug.Log("Robotic arm starts picking up part 6...");
                break;
            case 6:
                Debug.Log("Robotic arm starts picking up part 7...");
                break;
            case 7:
                Debug.Log("Robotic arm starts picking up part 8...");
                break;
            case 8:
                Debug.Log("Robotic arm starts picking up part 9...");
                break;
            case 9:
                Debug.Log("Robotic arm starts picking up part 10...");
                break;
            case 10:
                Debug.Log("Robotic arm starts picking up part 11...");
                break;
            case 11:
                Debug.Log("Robotic arm starts picking up part 12...");
                break;
            case 12:
                Debug.Log("Robotic arm starts picking up part 13...");
                break;
            case 13:
                Debug.Log("Robotic arm starts picking up part 14...");
                break;
            default:
                Debug.Log("Default operation for the robotic arm...");
                break;
        }

        // Add specific logic for controlling the robotic arm here
        // For example: send control signals to the robotic arm system
    }
}

// Class used to parse the JSON into a dictionary
[System.Serializable]
public class PartStatusWrapper
{
    public List<PartStatusItem> items;

    public Dictionary<string, bool> ToDictionary()
    {
        Dictionary<string, bool> dict = new Dictionary<string, bool>();
        foreach (var item in items)
        {
            dict.Add(item.key, item.value);
        }
        return dict;
    }
}

[System.Serializable]
public class PartStatusItem
{
    public string key;
    public bool value;
}

*/


/*************** Test on local ***************/

using UnityEngine;
using UnityEngine.UI;

public class AfterClick : MonoBehaviour
{
    // Array of button objects
    public Button[] buttons;

    // Robotic arm controller class or object
    public RoboticArmController roboticArm;

    // Set the disabled color
    public Color disabledColor = Color.gray;

    // Array of Image components for the buttons
    private Image[] buttonImages;

    void Start()
    {
        // Initialize the button Image components array
        buttonImages = new Image[buttons.Length];

        for (int i = 0; i < buttons.Length; i++)
        {
            int index = i; // Capture the current index
            buttonImages[i] = buttons[i].GetComponent<Image>();

            // Ensure the button reference is not null
            if (buttons[i] != null)
            {
                // Add a listener for each button
                buttons[i].onClick.AddListener(() => OnButtonClick(index));
            }
        }
    }

    // Function triggered when a button is clicked
    public void OnButtonClick(int buttonIndex)
    {
        Debug.Log("Button " + (buttonIndex + 1) + " is clicked to start picking up items...");

        // If the robotic arm reference is not null, invoke the pick-up operation
        if (roboticArm != null)
        {
            roboticArm.PickUpItem(buttonIndex); // Pass the button index as the command
        }

        // Disable the button to prevent duplicate clicks
        buttons[buttonIndex].interactable = false;

        // Set the button color to the disabled color
        if (buttonImages[buttonIndex] != null)
        {
            buttonImages[buttonIndex].color = disabledColor;
        }
    }
}

// Robotic arm control class, assuming this is the logic for controlling the robotic arm
public class RoboticArmController : MonoBehaviour
{
    // Perform different operations based on the button index
    public void PickUpItem(int buttonIndex)
    {
        switch (buttonIndex)
        {
            case 0:
                Debug.Log("Robotic arm starts picking up part 1...");
                break;
            case 1:
                Debug.Log("Robotic arm starts picking up part 2...");
                break;
            case 2:
                Debug.Log("Robotic arm starts picking up part 3...");
                break;
            case 3:
                Debug.Log("Robotic arm starts picking up part 4...");
                break;
            case 4:
                Debug.Log("Robotic arm starts picking up part 5...");
                break;
            case 5:
                Debug.Log("Robotic arm starts picking up part 6...");
                break;
            case 6:
                Debug.Log("Robotic arm starts picking up part 7...");
                break;
            case 7:
                Debug.Log("Robotic arm starts picking up part 8...");
                break;
            case 8:
                Debug.Log("Robotic arm starts picking up part 9...");
                break;
            case 9:
                Debug.Log("Robotic arm starts picking up part 10...");
                break;
            case 10:
                Debug.Log("Robotic arm starts picking up part 11...");
                break;
            case 11:
                Debug.Log("Robotic arm starts picking up part 12...");
                break;
            case 12:
                Debug.Log("Robotic arm starts picking up part 13...");
                break;
            case 13:
                Debug.Log("Robotic arm starts picking up part 14...");
                break;
            default:
                Debug.Log("Default operation for the robotic arm...");
                break;
        }
        // Add specific robotic arm control logic here
    }
}

