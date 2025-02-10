import cv2
import torch
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import argparse

class YOLODetector:
    """
    YOLO Detector using a model from Ultralytics (e.g. yolov5s).
    It detects objects in an image and filters out only the specified labels.
    """

    def __init__(self, model_weight='yolo11m.pt', device=None, track_labels=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load the YOLO model
        self.model = YOLO(model_weight).to(device)
        self.device = device
        # If no labels are specified, default to 'car'
        if track_labels is None:
            track_labels = ['car']
        # Normalize labels to lowercase for case-insensitive matching
        self.track_labels = [label.lower() for label in track_labels]

    def detect(self, frame):
        """
        Perform detection on the frame and return a list of detections for the specified labels.
        Each detection is a list:
            [x1, y1, x2, y2, confidence, label]
        """
        # Run inference
        results = self.model(frame, imgsz=640, conf=0.5, device=self.device)[0].boxes.cpu()

        xyxys = results.xyxy.tolist()
        confs = results.conf.tolist()
        clss = results.cls.tolist()
        filtered = []
        for box, conf, cls in zip(xyxys, confs, clss):
            x1, y1, x2, y2 = box
            # Get the class name from the model's names list
            label = self.model.names[int(cls)]
            # Filter detections: only keep those with a label in the track list
            if label.lower() in self.track_labels:
                filtered.append([x1, y1, x2, y2, conf, label])
        return filtered


class DeepSortTracker:
    """
    Wrapper for DeepSORT tracker.
    """

    def __init__(self):
        # Initialize the DeepSORT tracker.
        # You can adjust parameters like max_age, n_init, etc., as needed.
        self.tracker = DeepSort(max_age=30)

    def update(self, detections, frame):
        """
        Update the tracker with the new detections.
        Each detection should be a list: [x1, y1, x2, y2, confidence, label].
        DeepSORT expects detections in the format:
            ([x, y, width, height], confidence, label)
        """
        ds_detections = []
        for det in detections:
            x1, y1, x2, y2, conf, label = det
            width = x2 - x1
            height = y2 - y1
            ds_detections.append(([x1, y1, width, height], conf, label))

        # Update tracker: returns a list of track objects
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)
        return tracks


class VideoProcessor:
    """
    Video processing class that captures frames from a video source,
    performs detection and tracking, and displays the results.
    """

    def __init__(self, video_source, detector, tracker):
        """
        video_source: path to video file or integer (for webcam).
        detector: an instance of YOLODetector.
        tracker: an instance of a tracker (e.g., DeepSortTracker).
        """
        self.video_source = video_source
        self.detector = detector
        self.tracker = tracker

    def process(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 30
        cv2.namedWindow("Object Detection and Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Detection and Tracking", 1280, 720)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects in the frame
            detections = self.detector.detect(frame)
            # Update the tracker with the current detections
            tracks = self.tracker.update(detections, frame)

            # Draw results on the frame
            for track in tracks:
                # Only process confirmed tracks.
                if not track.is_confirmed():
                    continue

                # Get the track's ID and bounding box.
                track_id = track.track_id
                # to_ltrb() returns bounding box as [left, top, right, bottom]
                ltrb = track.to_ltrb()
                bbox = list(map(int, ltrb))

                # Get detection confidence and label (if provided by DeepSORT)
                conf = track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.0
                if conf is None:
                    conf = 0.0
                label = track.get_class() if hasattr(track, 'get_class') else "object"

                # Draw bounding box and label text
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              color=(0, 255, 0), thickness=4)
                text = f"{label} {conf:.2f} ID:{track_id}"
                font_scale = 1.0  # Adjust as needed
                font_thickness = 2  # Adjust as needed
                cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

            cv2.imshow("Object Detection and Tracking", frame)
            # Press 'q' to exit.
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Object Detection and Tracking")
    parser.add_argument("--video", type=str, default="0",
                        help="Path to video file or camera index (default: 0)")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="YOLO model weight to use (default: yolo11n.pt)")
    parser.add_argument("--labels", type=str, default="car",
                        help="Comma-separated list of object labels to track (default: 'car')")
    args = parser.parse_args()

    # Determine video source (integer if webcam, else string path)
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video

    # Process the labels argument (comma-separated list)
    track_labels = [label.strip() for label in args.labels.split(",") if label.strip()]

    # Initialize detector with specified labels
    detector = YOLODetector(model_weight=args.model, track_labels=track_labels)

    # Initialize tracker (using DeepSORT)
    tracker = DeepSortTracker()

    # Initialize and run the video processor.
    processor = VideoProcessor(video_source, detector, tracker)
    print("Starting video processing. Press 'q' to exit.")
    start_time = time.time()
    processor.process()
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
