import cv2
import sys

def test_camera():
    print("Testing Camera Access...")
    
    # Try different indices if 0 doesn't work
    for index in [0, 1, 2]:
        print(f"Trying camera index: {index}...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            print(f"Failed to open camera at index {index}")
            continue
            
        print(f"Camera {index} opened successfully! Reading first frame...")
        success, frame = cap.read()
        
        if success:
            print(f"Successfully grabbed a frame from camera {index}!")
            cv2.imshow(f"Camera {index} Test - Press any key to close", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cap.release()
            return True
        else:
            print(f"Failed to read from camera {index}")
        
        cap.release()
        
    print("\n[ERROR] No working camera found. Possible reasons:")
    print("1. Another app (Zoom, Chrome, etc.) is using your camera.")
    print("2. Camera is disabled in Windows Privacy Settings.")
    print("3. No camera drivers installed.")
    return False

if __name__ == "__main__":
    test_camera()
