import numpy as np
from scipy.stats import multivariate_normal
from plasticorigins.tracking.utils import in_frame, exp_and_normalise, GaussianMixture
from pykalman import KalmanFilter, AdditiveUnscentedKalmanFilter
import matplotlib.patches as mpatches

class Tracker:

    def __init__(self, frame_nb, X0, transition_variance, observation_variance, delta):

        self.transition_covariance = np.diag(transition_variance)
        self.observation_covariance = np.diag(observation_variance)
        self.updated = False
        self.steps_since_last_observation = 0
        self.enabled = True
        self.tracklet = [(frame_nb, X0)]
        self.delta = delta

    def store_observation(self, observation, frame_nb):
        self.tracklet.append((frame_nb, observation))
        self.updated = True

    def update_status(self, flow):
        if self.enabled and not self.updated:
            self.steps_since_last_observation += 1
            self.enabled = self.update(None, flow)
        else:
            self.steps_since_last_observation = 0
        self.updated = False

    def build_confidence_function(self, flow):

        def confidence_from_multivariate_distribution(coord, distribution):
            delta = self.delta
            x = coord[0]
            y = coord[1]
            right_top = np.array([x+delta, y+delta])
            left_low = np.array([x-delta, y-delta])
            right_low = np.array([x+delta, y-delta])
            left_top = np.array([x-delta, y+delta])

            return distribution.cdf(right_top) \
                - distribution.cdf(right_low) \
                - distribution.cdf(left_top) \
                + distribution.cdf(left_low)

        distribution = self.predictive_distribution(flow)

        return lambda coord: confidence_from_multivariate_distribution(coord, distribution)

    def get_display_colors(self, display, tracker_nb):
        colors = display.colors
        color = colors[tracker_nb % len(colors)]
        display.legends.append(mpatches.Patch(color=color, label=len(self.tracklet)))
        return colors[tracker_nb % len(colors)]

class SMC(Tracker):
    def set_param(param):
        SMC.n_particles = int(param)

    def __init__(self, frame_nb, X0, transition_variance, observation_variance, delta):
        super().__init__(frame_nb, X0, transition_variance, observation_variance, delta)

        self.particles = multivariate_normal(
            X0, cov=self.observation_covariance).rvs(SMC.n_particles)
        self.normalized_weights = np.ones(SMC.n_particles)/SMC.n_particles

    def update(self, observation, flow, frame_nb=None):
        if observation is not None: self.store_observation(observation, frame_nb)
        self.resample()
        enabled = self.move_particles(flow)
        if observation is not None:
            self.importance_reweighting(observation)
        else:
            self.normalized_weights = np.ones(
                len(self.particles))/len(self.particles)

        return enabled

    def state_transition(self, state, flow):

        mean = state + \
            flow[max(0, int(state[1])),
                 max(0, int(state[0])), :]
        cov = np.diag(self.transition_covariance)
        return multivariate_normal(mean, cov)

    def observation(self, state):

        return multivariate_normal(state, self.observation_covariance)

    def move_particles(self, flow):
        new_particles = []
        for particle in self.particles:
            new_particle = self.state_transition(particle, flow).rvs(1)
            if in_frame(new_particle, flow.shape[:-1]):
                new_particles.append(new_particle)
        if len(new_particles):
            self.particles = np.array(new_particles)
            enabled = True
        else:
            enabled = False

        return enabled

    def importance_reweighting(self, observation):
        log_weights_unnormalized = np.zeros(len(self.particles))
        for particle_nb, particle in enumerate(self.particles):
            log_weights_unnormalized[particle_nb] = self.observation(
                particle).logpdf(observation)
        self.normalized_weights = exp_and_normalise(log_weights_unnormalized)

    def resample(self):
        resampling_indices = np.random.choice(
            a=len(self.particles), p=self.normalized_weights, size=len(self.particles))
        self.particles = self.particles[resampling_indices]

    def predictive_distribution(self, flow, nb_new_particles=5):
        new_particles = []
        new_weights = []

        for particle, normalized_weight in zip(self.particles, self.normalized_weights):
            new_particles_for_particle = self.state_transition(
                particle, flow).rvs(nb_new_particles)

            new_particles_for_particle = [
                particle for particle in new_particles_for_particle if in_frame(particle, flow.shape[:-1])]

            if len(new_particles_for_particle):
                new_particles.extend(new_particles_for_particle)
                new_weights.extend([normalized_weight/len(new_particles_for_particle)] *
                                len(new_particles_for_particle))

        new_particles = np.array(new_particles)



        return GaussianMixture(new_particles, self.observation_covariance, new_weights)

    def fill_display(self, display, tracker_nb):
        color = self.get_display_colors(display, tracker_nb)
        display.ax.scatter(self.particles[:,0], self.particles[:,1], s=5, c=color)

class EKF(Tracker):

    def __init__(self, frame_nb, X0, transition_variance, observation_variance, delta):
            super().__init__(frame_nb, X0, transition_variance, observation_variance, delta)
            self.filter = KalmanFilter(initial_state_mean=X0,
                                       initial_state_covariance=self.observation_covariance,
                                       transition_covariance=self.transition_covariance,
                                       observation_matrices=np.eye(2),
                                       observation_covariance=self.observation_covariance)

            self.filtered_state_mean = X0
            self.filtered_state_covariance = self.observation_covariance

    def get_update_parameters(self, flow):

        flow_value = flow[int(self.filtered_state_mean[1]),int(self.filtered_state_mean[0]), :]


        grad_flow_value = np.array([np.gradient(flow[:,:,0]),np.gradient(flow[:,:,1])])[:,:,int(self.filtered_state_mean[1]),int(self.filtered_state_mean[0])]
        return np.eye(2) + grad_flow_value, flow_value - grad_flow_value.dot(self.filtered_state_mean)


    def EKF_step(self, observation, flow):
        transition_matrix, transition_offset = self.get_update_parameters(flow)

        return self.filter.filter_update(self.filtered_state_mean,
                                        self.filtered_state_covariance,
                                        transition_matrix=transition_matrix,
                                        transition_offset=transition_offset,
                                        observation=observation)

    def update(self, observation, flow, frame_nb=None):
        if observation is not None: self.store_observation(observation, frame_nb)

        self.filtered_state_mean, self.filtered_state_covariance = self.EKF_step(observation, flow)

        enabled=False if not in_frame(self.filtered_state_mean,flow.shape[:-1]) else True

        return enabled

    def predictive_distribution(self, flow):

        filtered_state_mean, filtered_state_covariance = self.EKF_step(None, flow)

        distribution = multivariate_normal(filtered_state_mean, filtered_state_covariance + self.observation_covariance)

        return distribution

    def fill_display(self, display, tracker_nb):
        yy, xx = np.mgrid[0:display.display_shape[1]:1, 0:display.display_shape[0]:1]
        pos = np.dstack((xx, yy))
        distribution = multivariate_normal(self.filtered_state_mean, self.filtered_state_covariance)

        color = self.get_display_colors(display, tracker_nb)
        cs = display.ax.contour(distribution.pdf(pos), colors=color)
        display.ax.clabel(cs, inline=True, fontsize='large')
        display.ax.scatter(self.filtered_state_mean[0], self.filtered_state_mean[1], color=color, marker="x", s=100)

class UKF(Tracker):

    def __init__(self, frame_nb, X0, transition_variance, observation_variance, delta):
            super().__init__(frame_nb, X0, transition_variance, observation_variance, delta)
            self.filter = AdditiveUnscentedKalmanFilter(initial_state_mean=X0,
                                        initial_state_covariance=self.observation_covariance,
                                        observation_functions = lambda z: np.eye(2).dot(z),
                                        transition_covariance=self.transition_covariance,
                                        observation_covariance=self.observation_covariance)

            self.filtered_state_mean = X0
            self.filtered_state_covariance = self.observation_covariance

    def UKF_step(self, observation, flow):
        return self.filter.filter_update(self.filtered_state_mean,
                                        self.filtered_state_covariance,
                                        transition_function=lambda x: x + flow[int(x[1]),int(x[0]),:],
                                        observation=observation)


    def update(self, observation, flow, frame_nb=None):
        if observation is not None: self.store_observation(observation, frame_nb)

        self.filtered_state_mean, self.filtered_state_covariance = self.UKF_step(observation, flow)

        enabled=False if not in_frame(self.filtered_state_mean,flow.shape[:-1]) else True

        return enabled

    def predictive_distribution(self, flow):

        filtered_state_mean, filtered_state_covariance = self.UKF_step(None, flow)

        distribution = multivariate_normal(filtered_state_mean, filtered_state_covariance + self.observation_covariance)

        return distribution

    def fill_display(self, display, tracker_nb):
        yy, xx = np.mgrid[0:display.display_shape[1]:1, 0:display.display_shape[0]:1]
        pos = np.dstack((xx, yy))
        distribution = multivariate_normal(self.filtered_state_mean, self.filtered_state_covariance)

        color = self.get_display_colors(display, tracker_nb)
        cs = display.ax.contour(distribution.pdf(pos), colors=color)
        display.ax.clabel(cs, inline=True, fontsize='large')
        display.ax.scatter(self.filtered_state_mean[0], self.filtered_state_mean[1], color=color, marker="x", s=100)

trackers = {'EKF': EKF,
           'SMC': SMC,
           'UKF': UKF}

def get_tracker(algorithm_and_params):
    print(f'{algorithm_and_params} will be used for tracking.')

    splitted_name = algorithm_and_params.split('_')
    if len(splitted_name) > 1:
        algorithm_name, param = splitted_name
        tracker = trackers[algorithm_name]
        tracker.set_param(param)

    else:
        algorithm_name = splitted_name[0]
        tracker = trackers[algorithm_name]

    return tracker



