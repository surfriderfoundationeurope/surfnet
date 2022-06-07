import matplotlib.patches as mpatches
import numpy as np
from pykalman import KalmanFilter
from scipy.stats import multivariate_normal

from plasticorigins.tracking.utils import in_frame


class Tracker:
    def __init__(
        self,
        frame_nb,
        X0,
        confidence,
        class_id,
        transition_variance,
        observation_variance,
        delta,
    ):
        self.transition_covariance = np.diag(transition_variance)
        self.observation_covariance = np.diag(observation_variance)
        self.updated = False
        self.steps_since_last_observation = 0
        self.enabled = True
        self.tracklet = [(frame_nb, X0, confidence, class_id)]
        self.delta = delta

    def store_observation(self, observation, frame_nb, confidence, class_id):
        self.tracklet.append((frame_nb, observation, confidence, class_id))
        self.updated = True

    def update_status(self, flow):
        if self.enabled and not self.updated:
            self.steps_since_last_observation += 1
            self.enabled = self.update(None, None, None, flow)
        else:
            self.steps_since_last_observation = 0
        self.updated = False

    def build_confidence_function(self, flow):
        def confidence_from_multivariate_distribution(coord, distribution):
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

    def cls_score_function(self, conf, label):
        """generates a score based on classes associated with observation in this tracker"""
        class_conf = sum(tr[2] for tr in self.tracklet if tr[3] == label)
        other_conf = sum(tr[2] for tr in self.tracklet)
        return (class_conf + conf) / (other_conf + conf)

    def get_display_colors(self, display, tracker_nb):
        colors = display.colors
        color = colors[tracker_nb % len(colors)]
        display.legends.append(mpatches.Patch(color=color, label=len(self.tracklet)))
        return colors[tracker_nb % len(colors)]


class EKF(Tracker):
    def __init__(
        self,
        frame_nb,
        X0,
        confidence,
        class_id,
        transition_variance,
        observation_variance,
        delta,
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

    def get_update_parameters(self, flow):

        flow_value = flow[
            int(self.filtered_state_mean[1]), int(self.filtered_state_mean[0]), :,
        ]

        grad_flow_value = np.array(
            [np.gradient(flow[:, :, 0]), np.gradient(flow[:, :, 1])]
        )[:, :, int(self.filtered_state_mean[1]), int(self.filtered_state_mean[0]),]
        return (
            np.eye(2) + grad_flow_value,
            flow_value - grad_flow_value.dot(self.filtered_state_mean),
        )

    def EKF_step(self, observation, flow):
        transition_matrix, transition_offset = self.get_update_parameters(flow)

        return self.filter.filter_update(
            self.filtered_state_mean,
            self.filtered_state_covariance,
            transition_matrix=transition_matrix,
            transition_offset=transition_offset,
            observation=observation,
        )

    def update(self, observation, confidence, class_id, flow, frame_nb=None):
        if observation is not None:
            self.store_observation(observation, frame_nb, confidence, class_id)

        (self.filtered_state_mean, self.filtered_state_covariance,) = self.EKF_step(
            observation, flow
        )

        enabled = (
            False if not in_frame(self.filtered_state_mean, flow.shape[:-1]) else True
        )

        return enabled

    def predictive_distribution(self, flow):

        filtered_state_mean, filtered_state_covariance = self.EKF_step(None, flow)

        distribution = multivariate_normal(
            filtered_state_mean,
            filtered_state_covariance + self.observation_covariance,
            allow_singular=True,
        )

        return distribution

    def fill_display(self, display, tracker_nb):
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


def get_tracker(algorithm_and_params):
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
