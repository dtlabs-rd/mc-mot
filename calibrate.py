import torch
import numpy as np
import cv2

import utilities


def main(opts):
    video1 = cv2.VideoCapture(opts.video1)
    assert video1.isOpened(), f"Could not open video1 source {opts.video1}"
    video2 = cv2.VideoCapture(opts.video2)
    assert video2.isOpened(), f"Could not open video2 source {opts.video2}"

    # 1000 was choosen arbitrarily
    feat_detector = cv2.SIFT_create(1000)

    _, frame1 = video1.read()
    _, frame2 = video2.read()

    kpts1, des1 = feat_detector.detectAndCompute(frame1, None)
    kpts2, des2 = feat_detector.detectAndCompute(frame2, None)

    bf = cv2.BFMatcher()

    # NOTE: k=2 means the euclidian distance between the two closest matches
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    cam4_H_cam1, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    np.save(f"{opts.homography_pth}.npy", cam4_H_cam1)

    src_pts = np.int0(src_pts).reshape(-1, 2)
    dst_pts = np.int0(dst_pts).reshape(-1, 2)

    img_with_matches = utilities.draw_matches(frame1, src_pts, frame2, dst_pts, good)

    assert cv2.imwrite("./img_with_matches.png", img_with_matches)

    # Loading yolov5 model
    detector = torch.hub.load("ultralytics/yolov5", "yolov5m")

    # NOTE: Avoid detecting multiple objects in the same box
    detector.agnostic = True
    detector.classes = [0]

    num_frames1 = video1.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames2 = video2.get(cv2.CAP_PROP_FRAME_COUNT)
    num_frames = min(num_frames2, num_frames1)
    num_frames = int(num_frames)

    # NOTE: Second video is 17 frames behind the first video
    video2.set(cv2.CAP_PROP_POS_FRAMES, 17)

    for idx in range(num_frames):
        # Get frames
        frame1 = video1.read()[1]
        frame2 = video2.read()[1]

        # Run object detection
        anno = detector([frame1, frame2])

        pred1 = anno.xyxy[0].cpu().numpy()[:, :4]
        pred2 = anno.xyxy[1].cpu().numpy()[:, :4]

        pred2_ = utilities.apply_homography_xyxy(pred1, cam4_H_cam1)

        utilities.draw_bounding_boxes(frame1, pred1)
        utilities.draw_bounding_boxes(frame2, pred2)
        utilities.draw_bounding_boxes(frame2, pred2_, color=(0, 0, 255))

        vis = np.concatenate([frame1, frame2], axis=1)

        cv2.namedWindow("vis", cv2.WINDOW_NORMAL)
        cv2.imshow("vis", vis)
        key = cv2.waitKey(0)

        if key == ord("q"):
            break

    video1.release()
    video2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video1", type=str, help="Path to the video cam1.mp4.")
    parser.add_argument("--video2", type=str, help="Path to the video cam4.mp4.")
    parser.add_argument(
        "--homography-pth",
        type=str,
        help="Path to save the homography as a npy file. You don't have to include a .npy at the end, just the name of the file.",
    )

    opts = parser.parse_args()

    main(opts)
