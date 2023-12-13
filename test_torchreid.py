import torchreid
import torch
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


extractor = torchreid.utils.FeatureExtractor(
    model_name="osnet_x1_0", model_path="./weights/osnet_x1_0.pth.tar", device="cuda"
)

video1 = cv2.VideoCapture(
    "/home/mbenencase/projects/daedalus/datasets/multi-camera-tracking/epfl/cam1.mp4"
)
assert video1.isOpened(), "Could not open video1"
video2 = cv2.VideoCapture(
    "/home/mbenencase/projects/daedalus/datasets/multi-camera-tracking/epfl/cam4.mp4"
)
assert video2.isOpened(), "Could not open video2"

# Loading yolov5 model
detector = torch.hub.load("ultralytics/yolov5", "yolov5m")
detector.agnostic = True
detector.classes = [0]
detector.conf = 0.5

num_frames = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))

COS_THRES: float = 0.80
COLORS = np.random.randint(0, 255, size=(100, 3), dtype="uint8")

video = None
for idx in range(num_frames):
    # Get frames
    frame1 = video1.read()[1]
    frame2 = video2.read()[1]

    # Run object detection
    anno = detector([frame1, frame2])

    preds1 = anno.xyxy[0].cpu().numpy()
    preds2 = anno.xyxy[1].cpu().numpy()

    cam1_features = []
    cam2_features = []
    for pred in preds1:
        x1, y1, x2, y2, _, _ = np.int0(pred)
        crop = frame1[y1:y2, x1:x2, :]

        feat = extractor(crop)[0].cpu().numpy()
        feat = feat / np.linalg.norm(feat)
        cam1_features.append(feat)
    for pred in preds2:
        x1, y1, x2, y2, _, _ = np.int0(pred)
        crop = frame2[y1:y2, x1:x2, :]

        feat = extractor(crop)[0].cpu().numpy()
        feat = feat / np.linalg.norm(feat)
        cam2_features.append(feat)

    cam1_features = np.array(cam1_features)
    cam2_features = np.array(cam2_features)

    sim_matrix = cam1_features @ cam2_features.T
    matched_indices = linear_assignment(-sim_matrix)

    for idx, match in enumerate(matched_indices):
        if sim_matrix[match[0], match[1]] < COS_THRES:
            continue
        else:
            # Draw bounding boxes
            x1, y1, x2, y2, _, _ = np.int0(preds1[match[0]])
            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame1,
                f"{idx}",
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            x1, y1, x2, y2, _, _ = np.int0(preds2[match[1]])
            cv2.rectangle(frame2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame2, f"{idx}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

    vis = np.hstack([frame1, frame2])

    if video is None:
        H, W, _ = vis.shape
        video = cv2.VideoWriter(
            "./reid.avi",
            cv2.VideoWriter_fourcc(*"MJPG"),
            15,
            (W, H),
            True,
        )

    video.write(vis)

    cv2.namedWindow("Vis", cv2.WINDOW_NORMAL)
    cv2.imshow("Vis", vis)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video1.release()
video2.release()
video.release()
cv2.destroyAllWindows()
