import numpy as np
from sort import associate_detections_to_trackers


def modify_bbox_source(bboxes, homography):
    """
    Modify the source of bounding boxes.

    Args:
        bboxes (np.ndarray): Bounding boxes to modify.
        H (np.ndarray): Homography matrix.
    Returns:
        np.ndarray: Modified bounding boxes.
    """

    bboxes_ = list()

    for bbox in bboxes:
        x0, y0, x1, y1, *keep = bbox

        p0 = np.dot(homography, np.array([x0, y0, 1]).reshape(3, 1)).reshape(3)
        p1 = np.dot(homography, np.array([x1, y1, 1]).reshape(3, 1)).reshape(3)

        x0 = int(p0[0] / p0[-1])
        y0 = int(p0[1] / p0[-1])
        x1 = int(p1[0] / p1[-1])
        y1 = int(p1[1] / p1[-1])
        bboxes_.append([x0, y0, x1, y1] + keep)

    return np.asarray(bboxes_)


class MultiCameraTracker:
    def __init__(self, homographies: list, iou_thres=0.2):
        """
        Multi Camera Tracking class contructor.
        """
        self.num_sources = len(homographies)
        self.homographies = homographies
        self.iou_thres = iou_thres
        self.next_id = 1

        self.ids = [{} for _ in range(self.num_sources)]
        self.age = [{} for _ in range(self.num_sources)]

    def update(self, tracks: list):
        # Project tracks to a common reference
        proj_tracks = []
        for i, trks in enumerate(tracks):
            proj_tracks.append(modify_bbox_source(trks, self.homographies[i]))

        # For each pair of sources
        for i in range(self.num_sources):
            for j in range(i + 1, self.num_sources):
                # Match tracks with IOU
                matched = {}
                matches, unmatches_i, unmatches_j = associate_detections_to_trackers(
                    proj_tracks[i], proj_tracks[j], iou_threshold=self.iou_thres
                )

                # Set global ids for the matched tracks
                for idx_i, idx_j in matches:
                    # Ids
                    id_i = proj_tracks[i][idx_i][4]
                    id_j = proj_tracks[j][idx_j][4]
                    # Current match ids
                    match_i = self.ids[i].get(id_i)
                    match_j = self.ids[j].get(id_j)

                    # If track i has a global id and is older then track j
                    if (
                        match_i != None
                        and self.age[i].get(id_i, 0) >= self.age[j].get(id_j, 0)
                        and not matched.get(match_i, False)
                    ):
                        self.ids[j][id_j] = match_i
                        matched[match_i] = True
                    # Else if track j has a global id
                    elif match_j != None and not matched.get(match_j, False):
                        self.ids[i][id_i] = match_j
                        matched[match_j] = True
                    # None of them has a global id
                    else:
                        self.ids[i][id_i] = self.next_id
                        self.ids[j][id_j] = self.next_id
                        matched[self.next_id] = True
                        self.next_id += 1

                    # Increment tracks age
                    self.age[i][id_i] = self.age[i].get(id_i, 0) + 1
                    self.age[j][id_j] = self.age[j].get(id_j, 0) + 1

                # Set global ids for unmatched tracks
                for idx_i in unmatches_i:
                    id_i = proj_tracks[i][idx_i][4]
                    match_i = self.ids[i].get(id_i)

                    if match_i == None or matched.get(match_i, False):
                        self.ids[i][id_i] = self.next_id
                        matched[self.next_id] = True
                        self.next_id += 1

                    # Increment track age
                    self.age[i][id_i] = self.age[i].get(id_i, 0) + 1

                for idx_j in unmatches_j:
                    id_j = proj_tracks[j][idx_j][4]
                    match_j = self.ids[j].get(id_j)

                    if match_j == None or matched.get(match_j, False):
                        self.ids[j][id_j] = self.next_id
                        matched[self.next_id] = True
                        self.next_id += 1

                    # Increment track age
                    self.age[j][id_j] = self.age[j].get(id_j, 0) + 1

        return self.ids
