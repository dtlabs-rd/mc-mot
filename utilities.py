import cv2
import numpy as np


centroids = {}


def apply_homography(uv, H):
    uv_ = np.zeros_like(uv)

    for idx, (u, v) in enumerate(uv):
        uvs = H @ np.array([u, v, 1]).reshape(3, 1)
        u_, v_, s_ = uvs.reshape(-1)
        u_ = u_ / s_
        v_ = v_ / s_

        uv_[idx] = [u_, v_]

    return uv_


def apply_homography_xyxy(xyxy, H):
    xyxy_ = np.zeros_like(xyxy)

    for idx, (x1, y1, x2, y2) in enumerate(xyxy):
        x1, y1, s1 = H @ np.array([x1, y1, 1]).reshape(3, 1)
        x1 = x1 / s1
        y1 = y1 / s1

        x2, y2, s2 = H @ np.array([x2, y2, 1]).reshape(3, 1)
        x2 = x2 / s2
        y2 = y2 / s2

        xyxy_[idx] = [x1, y1, x2, y2]

    return xyxy_


def draw_bounding_boxes(image, bounding_boxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image given a list of (x1, y1, x2, y2) coordinates.

    :param image: The input image to draw the bounding boxes on.
    :type image: numpy.ndarray
    :param bounding_boxes: A list of (x1, y1, x2, y2) coordinates for each bounding box.
    :type bounding_boxes: list[tuple(int, int, int, int)]
    :param color: The color of the bounding boxes. Default is green.
    :type color: tuple(int, int, int)
    :param thickness: The thickness of the bounding boxes. Default is 2.
    :type thickness: int
    """

    for bbox in bounding_boxes:
        x1, y1, x2, y2 = np.int0(bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_matches(img1, kpts1, img2, kpts2, matches):
    vis = np.hstack([img1, img2])
    MAX_DIST_VAL = max([match.distance for match in matches])

    WIDTH = img2.shape[1]

    for idx, (src, dst, match) in enumerate(zip(kpts1, kpts2, matches)):
        src_x, src_y = src
        dst_x, dst_y = dst
        dst_x += WIDTH

        COLOR = (0, int(255 * (match.distance / MAX_DIST_VAL)), 0)

        vis = cv2.line(vis, (src_x, src_y), (dst_x, dst_y), COLOR, 1)

    return vis


def color_from_id(id):
    np.random.seed(id)
    return np.random.randint(0, 255, size=3).tolist()


def draw_tracks(image, tracks, ids_dict, src, classes=None):
    """
    Draw bounding boxes on an image and print tracking IDs for each box.

    Args:
        image: An array representing the image to draw on.
        boxes: A list of bounding boxes, where each box is a tuple of (x, y, w, h).
        ids: A list of tracking IDs, where each ID corresponds to a box in the boxes list.
    """
    # Convert the image to RGB color space
    vis = np.array(image)
    bboxes = tracks[:, :4]
    ids = tracks[:, 4]
    labels = tracks[:, 5]
    centroids[src] = centroids.get(src, {})

    # Loop over each bounding box and draw it on the image
    for i, box in enumerate(bboxes):
        id = ids_dict[ids[i]]
        color = color_from_id(id)

        # Get the box coordinates
        x1, y1, x2, y2 = np.int0(box)

        # Draw the box on the image
        if centroids == None:
            vis = cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=2)
        else:
            centroids[src][id] = centroids[src].get(id, [])
            centroids[src][id].append(((x1 + x2) // 2, (y1 + y2) // 2))
            vis = draw_history(vis, box, centroids[src][id], color)

        # Print the tracking ID next to the box
        if classes == None:
            text = f"{labels[i]} {id}"
        else:
            text = f"{classes[labels[i]]} {id}"
        vis = cv2.putText(
            vis, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2
        )

    return vis


def draw_label(image, x, y, label, track_id, color):
    # Convert the image to RGB color space
    vis = np.array(image)

    # Print the tracking ID next to the box
    text = f"{label} {track_id}"
    vis = cv2.putText(
        vis, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2
    )

    return vis


def draw_history(image, box, centroids, color):
    """
    Draw a bounding box and its historical centroids on an image.

    Args:
        image: An array representing the image to draw on.
        box: A tuple of (x, y, w, h) representing the bounding box to draw.
        centroids: A list of tuples representing the historical centroids of the bounding box.
    """
    # Convert the image to RGB color space
    vis = np.array(image)

    # Draw the bounding box on the image
    x1, y1, x2, y2 = np.int0(box)
    thickness = 2
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

    centroids = np.int0(centroids)
    # Draw the historical centroids on the vis
    for i, centroid in enumerate(centroids):
        if i == 0:
            # Draw the current centroid as a circle
            cv2.circle(vis, centroid, 2, color, thickness=-1)
        else:
            # Draw the historical centroids as lines connecting them
            prev_centroid = centroids[i - 1]
            cv2.line(vis, prev_centroid, centroid, color, thickness=2)

    return vis
