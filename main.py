import torch
import numpy as np
import cv2

import sort
import utilities
import homography_tracker


def main(opts):
    video1 = cv2.VideoCapture(opts.video1)
    assert video1.isOpened(), f"Could not open video1 source {opts.video1}"
    video2 = cv2.VideoCapture(opts.video2)
    assert video2.isOpened(), f"Could not open video2 source {opts.video2}"

    cam4_H_cam1 = np.load(opts.homography)
    cam1_H_cam4 = np.linalg.inv(cam4_H_cam1)

    homographies = list()
    homographies.append(np.eye(3))
    homographies.append(cam1_H_cam4)

    detector = torch.hub.load("ultralytics/yolov5", "yolov5m")
    detector.agnostic = True

    # Class 0 is Person
    detector.classes = [0]
    detector.conf = opts.conf

    trackers = [
        sort.Sort(
            max_age=opts.max_age, min_hits=opts.min_hits, iou_threshold=opts.iou_thres
        )
        for _ in range(2)
    ]
    global_tracker = homography_tracker.MultiCameraTracker(homographies, iou_thres=0.20)

    num_frames1 = video1.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames2 = video2.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames = min(num_frames2, num_frames1)
    num_frames = int(num_frames)

    # NOTE: Second video 'cam4.mp4' is 17 frames behind the first video 'cam1.mp4'
    video2.set(cv2.CAP_PROP_POS_FRAMES, 17)

    video = None
    for idx in range(num_frames):
        # Get frames
        frame1 = video1.read()[1]
        frame2 = video2.read()[1]

        # NOTE: YoloV5 expects the images to be RGB instead of BGR
        frames = [frame1[:, :, ::-1], frame2[:, :, ::-1]]

        anno = detector(frames)

        dets, tracks = [], []
        for i in range(len(anno)):
            # Sort Tracker requires (x1, y1, x2, y2) bounding box shape
            det = anno.xyxy[i].cpu().numpy()
            det[:, :4] = np.int0(det[:, :4])
            dets.append(det)

            # Updating each tracker measures
            tracker = trackers[i].update(det[:, :4], det[:, -1])
            tracks.append(tracker)

        global_ids = global_tracker.update(tracks)

        for i in range(2):
            frames[i] = utilities.draw_tracks(
                frames[i][:, :, ::-1],
                tracks[i],
                global_ids[i],
                i,
                classes=detector.names,
            )

        vis = np.hstack(frames)

        cv2.namedWindow("Vis", cv2.WINDOW_NORMAL)
        cv2.imshow("Vis", vis)
        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    video1.release()
    video2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video1", type=str, default="./epfl/cam1.mp4")
    parser.add_argument("--video2", type=str, default="./epfl/cam4.mp4")
    parser.add_argument("--homography", type=str, default="./cam4_H_cam1.npy")
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.3,
        help="IOU threshold to consider a match between two bounding boxes.",
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Max age of a track, i.e., how many frames will we keep a track alive.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Minimum number of matches to consider a track.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="Confidence value for the YoloV5 detector.",
    )

    opts = parser.parse_args()

    main(opts)
