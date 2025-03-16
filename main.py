from utils.realtime_extended import realtime_vdo


if __name__ == '__main__':

    # Show video from different type
    # realtime_vdo(case=0, color='rgb').get_vdo()       # show camera
    # realtime_vdo().get_screenshot1()                  # screenshot ver. 1
    # realtime_vdo(monitor_number=2).get_screenshot2()  # screenshot ver. 2

    # Detect object realtime
    realtime_vdo(monitor_number=2, model_name="yolov8n.pt").get_object_detection()    # detect object from screenshot ver. 1
    # realtime_vdo(monitor_number=2, model_name="yolov8n.pt").get_object_detection2()   # detect object from screenshot ver. 2


