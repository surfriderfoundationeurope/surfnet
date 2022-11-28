"""The ``trackers`` submodule provides several functions to build trackers for videos.

This submodule contains the following classes:
- ``Tracker`` : Build a Tracker for videos.
- ``EKF(Tracker)`` : Extended Kalman Filter based on Tracker class.

This submodule contains the following functions:
- ``get_tracker(algorithm_and_params:str)`` : Provide specific tracker according algorithm and parameters.

"""

import matplotlib.patches as mpatches
import numpy as np
from numpy import array, ndarray, dtype, float64
from typing import Any, Tuple
from cv2 import Mat
from pykalman import KalmanFilter
from scipy.stats import multivariate_normal

from plasticorigins.tracking.utils import in_frame


class Tracker:

    """Build a Tracker for videos.

    Args:
        frame_nb (int): the number of the frame
        X0 (array[Any,dtype[float64]]): Array of state mean
        confidence (float): confidence in [0,1]
        class_id (int): id of the object class
        transition_variance (ndarray[Any,dtype[float64]]): Array of the transition variance
        observation_variance (ndarray[Any,dtype[float64]]): Array of the observation variance
        delta (float): the probability threshold of an object to belong to a specific neighbourhood
    """

    def __init__(
        self,
        frame_nb: int,
        X0: array,
        confidence: float,
        class_id: int,
        transition_variance: ndarray[Any, dtype[float64]],
        observation_variance: ndarray[Any, dtype[float64]],
        delta: float,
    ):
        self.transition_covariance = np.diag(transition_variance)
        self.observation_covariance = np.diag(observation_variance)
        self.updated = False
        self.steps_since_last_observation = 0
        self.enabled = True
        self.tracklet = [(frame_nb, X0, confidence, class_id)]
        self.delta = delta

    def store_observation(
        self,
        observation: array,
        frame_nb: int,
        confidence: float,
        class_id: int,
    ) -> None:

        """Method to store observations from frames.

        Args:
            observation (array): observation of the current frame
            frame_nb (int): numero of the current frame
            confidence (float): confidence of an observation
            class_id (int): object class id
        """

        self.tracklet.append((frame_nb, observation, confidence, class_id))
        self.updated = True

    def update_status(self, flow: Mat) -> None:

        """Update status from input flow.

        Args:
            flow (Mat): the input flow for updating status
        """

        if self.enabled and not self.updated:
            self.steps_since_last_observation += 1
            self.enabled = self.update(None, None, None, flow)

        else:
            self.steps_since_last_observation = 0

        self.updated = False

    def build_confidence_function(self, flow: Mat) -> Mat:

        """Build confidence function from flow.

        Args:
            flow (array): the input flow

        Returns:
            The computing confidence distribution based on predictive distribution from flow.
        """

        def confidence_from_multivariate_distribution(
            coord: array, distribution: array
        ) -> Mat:

            """Computes confidence from multivariate distribution.

            Args:
                coord (array): coordinates of the local object
                distribution (array): mathematical input distribution

            Returns:
                The computing confidence distribution.
            """

            delta = self.delta
            x = coord[0]
            y = coord[1]
            right_top = np.array([x + delta, y + delta])
            left_low = np.array([x - delta, y - delta])
            right_low = np.array([x + delta, y - delta])
            left_top = np.array([x - delta, y + delta])

            return (
                distribution.cdf(right_top)
                - distribution.cdf(right_low)
                - distribution.cdf(left_top)
                + distribution.cdf(left_low)
            )

        distribution = self.predictive_distribution(flow)

        return lambda coord: confidence_from_multivariate_distribution(
            coord, distribution
        )

    def cls_score_function(self, conf: float, label: int) -> float:

        """Generates a score based on classes associated with observation in this tracker.

        Args:
            conf (float): confidence of the label
            label (int): label id

        Returns:
            A score based on classes associated with observation.
        """

        class_conf = sum(tr[2] for tr in self.tracklet if tr[3] == label)
        other_conf = sum(tr[2] for tr in self.tracklet)

        return (class_conf + conf) / (other_conf + conf)

    def get_display_colors(self, display: Any, tracker_nb: int) -> array:

        """Get the display colors for a tracker.

        Args:
            display (Any): display from tracker
            tracker_nb (int): the number of the current tracker

        Returns:
            colors (array) : The colors for the current tracker
        """

        colors = display.colors
        color = colors[tracker_nb % len(colors)]
        display.legends.append(mpatches.Patch(color=color, label=len(self.tracklet)))

        return colors[tracker_nb % len(colors)]


class EKF(Tracker):

    """Extended Kalman Filter : infinite impulse response filter that estimates the states of a dynamic system
    from a series of incomplete or noisy measurements.

    Args:
        frame_nb (int): the number of the frame
        X0 (array[Any,dtype[float64]]): Array of initial state mean
        confidence (float): confidence in [0,1]
        class_id (int): id of the object class
        transition_variance (ndarray[Any,dtype[float64]]): Array of the transition variance
        observation_variance (ndarray[Any,dtype[float64]]): Array of the observation variance
        delta (float): the probability threshold of an object to belong to a specific neighbourhood
    """

    def __init__(
        self,
        frame_nb: int,
        X0: array,
        confidence: float,
        class_id: int,
        transition_variance: ndarray[Any, dtype[float64]],
        observation_variance: ndarray[Any, dtype[float64]],
        delta: float,
    ):
        super().__init__(
            frame_nb,
            X0,
            confidence,
            class_id,
            transition_variance,
            observation_variance,
            delta,
        )
        self.filter = KalmanFilter(
            initial_state_mean=X0,
            initial_state_covariance=self.observation_covariance,
            transition_covariance=self.transition_covariance,
            observation_matrices=np.eye(2),
            observation_covariance=self.observation_covariance,
        )

        self.filtered_state_mean = X0
        self.filtered_state_covariance = self.observation_covariance

    def get_update_parameters(self, flow: Mat) -> Tuple[array, array]:

        """Update parameters from input flow.

        Args:
            flow (Mat): the input flow for updating status

        Returns:
            The updated flow value with gradient operation.
        """

        flow_value = flow[
            int(self.filtered_state_mean[1]),
            int(self.filtered_state_mean[0]),
            :,
        ]

        grad_flow_value = np.array(
            [np.gradient(flow[:, :, 0]), np.gradient(flow[:, :, 1])]
        )[
            :,
            :,
            int(self.filtered_state_mean[1]),
            int(self.filtered_state_mean[0]),
        ]
        return (
            np.eye(2) + grad_flow_value,
            flow_value - grad_flow_value.dot(self.filtered_state_mean),
        )

    def EKF_step(self, observation: array, flow: Mat) -> tuple:

        """Perform a one-step update to estimate the state at time ``t+1`` given an observation
        at time ``t+1`` and the previous estimate for time ``t`` given observations from times ``[0...t]``.
        This method is useful if one wants to track an object with streaming observations.

        Args:
            observation (array): the input observation for EKF at time ``t+1``
            flow (Mat): the input flow for EKF

        Returns:
            EKF_step (tuple) : Perform a one-step update to estimate the state at time ``t+1``
        """

        transition_matrix, transition_offset = self.get_update_parameters(flow)

        return self.filter.filter_update(
            self.filtered_state_mean,
            self.filtered_state_covariance,
            transition_matrix=transition_matrix,
            transition_offset=transition_offset,
            observation=observation,
        )

    def update(
        self,
        observation: array,
        confidence: float,
        class_id: int,
        flow: Mat,
        frame_nb: int = None,
    ) -> bool:

        """Enable the update according the EKF results from the current step.

        Args:
            observation (array): current observation of the frame
            confidence (float): confidence of the observation
            class_id (int): the object class in the observation
            flow (Mat): the input flow for updating status
            frame_nb (int): the numero of the frame. Set as default to ``None``

        Returns:
            enabled (bool): Enable the update if ``True``, not if ``False``
        """

        if observation is not None:
            self.store_observation(observation, frame_nb, confidence, class_id)

        (
            self.filtered_state_mean,
            self.filtered_state_covariance,
        ) = self.EKF_step(observation, flow)

        enabled = (
            False if not in_frame(self.filtered_state_mean, flow.shape[:-1]) else True
        )

        return enabled

    def predictive_distribution(self, flow: Mat) -> array:

        """Build predictive distribution from flow.

        Args:
            flow (Mat): the input flow

        Returns:
            The computed predictive distribution from normal multivariate model.
        """

        filtered_state_mean, filtered_state_covariance = self.EKF_step(None, flow)

        distribution = multivariate_normal(
            filtered_state_mean,
            filtered_state_covariance + self.observation_covariance,
            allow_singular=True,
        )

        return distribution

    def fill_display(self, display: Any, tracker_nb: int) -> None:

        """Fill display of a specific tracker.

        Args:
            display (Any): the display for tracker
            tracker_nb (int): the number of tracker
        """

        yy, xx = np.mgrid[
            0 : display.display_shape[1] : 1, 0 : display.display_shape[0] : 1
        ]
        pos = np.dstack((xx, yy))
        distribution = multivariate_normal(
            self.filtered_state_mean, self.filtered_state_covariance
        )

        color = self.get_display_colors(display, tracker_nb)
        cs = display.ax.contour(distribution.pdf(pos), colors=color)
        display.ax.clabel(cs, inline=True, fontsize="large")
        display.ax.scatter(
            self.filtered_state_mean[0],
            self.filtered_state_mean[1],
            color=color,
            marker="x",
            s=100,
        )


trackers = {"EKF": EKF}


def get_tracker(algorithm_and_params: str) -> Tracker:

    """Provide specific tracker according algorithm and parameters.

    Args:
        algorithm_and_params (str): specify the algorithm and the parameters used for tracker. You may specify only the algorithm.

    Returns:
        tracker (Tracker): Set the specific tracker according algorithm and parameters
    """

    print(f"{algorithm_and_params} will be used for tracking.")

    splitted_name = algorithm_and_params.split("_")
    if len(splitted_name) > 1:
        algorithm_name, param = splitted_name
        tracker = trackers[algorithm_name]
        tracker.set_param(param)

    else:
        algorithm_name = splitted_name[0]
        tracker = trackers[algorithm_name]

    return tracker
