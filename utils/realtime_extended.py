import cv2
import pyautogui
import mss
import numpy as np
from ultralytics import YOLO

class realtime_vdo():
    def __init__(self, case=0, color='rgb', monitor_number=1, model_name="yolov8n.pt"):
        self.case = case
        self.color = color
        self.cap = None
        self.monitor_number = monitor_number
        self.model_name = model_name
        self.frame = None

    def get_vdo(self):
        # 0 = camera
        # 1 = file
        # 3 = external source
        # 4 = screenshot use pyautogui simpler
        # 5 = screenshot use mss faster
        if self.case == 0:
            self.cap = cv2.VideoCapture(0)               # Open webcam (0 is usually the default camera)
            self.show_vdo()
        elif self.case == 1:
            self.cap = cv2.VideoCapture('video.mp4')     # If you have a video file, replace 0 with the file path:
            self.show_vdo()
        elif self.case == 2:
            ip_camera_url = "rtsp://username:password@camera_ip:port/stream"
            self.cap = cv2.VideoCapture(ip_camera_url)
            self.show_vdo()
        elif self.case == 3:
            gstream_pipeline = "your_gstreamer_pipeline_here"   # If you have video stream (e.g., from an external source), you can use GStreamer in OpenCV:
            self.cap = cv2.VideoCapture(gstream_pipeline, cv2.CAP_GSTREAMER)
            self.show_vdo()
        elif self.case == 4:
            self.get_screenshot1()
        elif self.case == 5:
            self.get_screenshot2()
        else:
            print('Invalid case input!')


    def show_vdo(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            if self.color == 'gray':
                # Process the frame (e.g., convert to grayscale)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Display the processed frame
                cv2.imshow('Video Stream', gray)

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            elif self.color == 'rgb':
                # Display the original color frame
                cv2.imshow('Video Stream', frame)

                # Press 'q' to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print('Invalid color input!')

        self.cap.release()
        cv2.destroyAllWindows()

    def get_screenshot1(self):
        while True:
            # Capture screen (as an image)
            screenshot = pyautogui.screenshot()

            # Convert PIL image to NumPy array
            frame = np.array(screenshot)

            # Convert RGB to BGR (OpenCV uses BGR format)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Display the captured screen
            cv2.imshow("Screen Capture", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def get_screenshot2(self):

        # Create a screen capture object
        sct = mss.mss()

        # Get screen dimensions
        monitor = sct.monitors[self.monitor_number]  # Use monitor[1] for the primary screen

        while True:
            # Capture screen
            screenshot = sct.grab(monitor)

            # Convert to NumPy array
            frame = np.array(screenshot)

            # Convert BGRA to BGR (remove alpha channel)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Display the screen capture
            cv2.imshow("Screen Capture", frame)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def get_object_detection(self):

        # Load the model eg,. YOLOv8
        model = YOLO(self.model_name)

        # Create a screen capture object
        sct = mss.mss()

        # Get screen dimensions
        monitor = sct.monitors[self.monitor_number]  # Use monitor[1] for the primary screen

        while True:
            # Capture screen
            screenshot = sct.grab(monitor)

            # Convert to NumPy array
            frame = np.array(screenshot)

            # Convert BGRA to BGR (remove alpha channel)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Run model on the frame : (frame, conf=0.2, iou=0.5, classes=[0, 2], agnostic_nms=True, max_det=500)
            results = model(frame, conf=0.2, iou=0.5, agnostic_nms=True, max_det=500)

            # Show the detection results
            for result in results:
                # Draw bounding boxes and labels
                frame = result.plot()  # This will add the boxes, labels, and scores to the frame

            # Display the frame with detections in a window
            cv2.imshow("Detection - Press 'q' to exit", frame)


            # # Show the detection results
            # for result in results:
            #     result.show()
            #
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
    def get_object_detection2(self):

        # Load the YOLOv8 model (e.g., YOLOv8 Large)
        model = YOLO(self.model_name)

        # Create a screen capture object
        sct = mss.mss()

        # Define the monitor number (e.g., 1 for the primary monitor)
        monitor = sct.monitors[self.monitor_number]

        # Start capturing the screen and running the model in a loop
        while True:
            # Capture the screen
            screenshot = sct.grab(monitor)

            # Convert to NumPy array
            frame = np.array(screenshot)

            # Convert BGRA to BGR (remove alpha channel)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Run the model on the frame (frame, conf=0.2, iou=0.5, classes=[0, 2], agnostic_nms=True, max_det=500)
            results = model(frame, conf=0.5, iou=0.5, agnostic_nms=True, max_det=500)

            # Draw the detection results (bounding boxes and labels)
            for result in results:
                for box in result.boxes:
                    # Get the class index and class name
                    class_idx = int(box.cls)
                    class_name = model.names[class_idx]

                    # Get bounding box coordinates and the confidence score
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    confidence = box.conf.item()  # Convert tensor to a regular float

                    # Draw the bounding box on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)  #  BGR

                    # Draw the label with the confidence score
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Show the frame with detection results
            cv2.imshow('Screen Capture with Object Detection', frame)

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        cv2.destroyAllWindows()


# if __name__ == "__main__":
    # realtime_vdo(case=0, color='rgb', monitor_number=2).get_vdo()
    # realtime_vdo(case=0, color='rgb', monitor_number=2).get_screenshot1()
    # realtime_vdo(case=0, color='rgb', monitor_number=2).get_screenshot2()
    # realtime_vdo(case=0, color='rgb', monitor_number=2, model_name="yolov8n.pt").get_object_detection()
    # realtime_vdo(case=0, color='rgb', monitor_number=2, model_name="yolov8n.pt").get_object_detection2()
