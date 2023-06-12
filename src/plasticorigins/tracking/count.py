from typing import List, Tuple, Union
import numpy as np
import os.path as op


def video_count_truth(video_path: str) -> int:
    """Get video object count from the videoname.txt file.

    Args:
        video_path (str): _description_

    Returns:
        int: number of objects.
    """
    video_count_file = op.splitext(video_path)[0] + ".txt"
    try:
        n = np.loadtxt(video_count_file)
        return int(n)
    except OSError as e:
        warning_msg = (
            "WARNING : Objects count is expected in the file "
            f"{video_count_file}. Make sure the file is available "
            "or set the compare argument to false."
        )

        print(warning_msg)
        raise e


def count_detected_objects(
    results: List[Tuple], video_path: str, compare: bool
) -> tuple[int, Union[int, None]]:
    """Evaluate the number of detected object.

    Args:
        results (List[Tuple]): raw filtered tracks
        video_path (str): video file path
        compare (bool): whether to compare to the manual count

    Returns:
        tuple[int, Union[int, None]]: number of detected objects and ground \
            truth count. If compare is false, the second term is None.
    """
    # number of detected object by counting unique object id.
    n_det = len(set(o[1] for o in results))

    n = None

    if compare:
        # nb of object in the video
        n = video_count_truth(video_path)

    return n_det, n
