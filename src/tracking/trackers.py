import numpy as np
from scipy.spatial.distance import euclidean 
from scipy.stats import multivariate_normal, norm
from tracking.utils import in_frame, exp_and_normalise, GaussianMixture, MultivariateDiscrete
from pykalman import KalmanFilter
import matplotlib.patches as mpatches


class Tracker:

    delta = None
    def __init__(self, frame_nb, X0, state_variance, observation_variance, delta):


        self.state_covariance = np.diag(state_variance)
        self.observation_covariance = np.diag(observation_variance)
        self.updated = False
        self.steps_since_last_observation = 0
        self.enabled = True
        self.tracklet = [(frame_nb, X0)]
        self.delta = delta

    def update(self, observation, frame_nb):
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
        # return lambda coord: euclidean(coord, distribution.mean)
    
    def fill_display(self, display, tracker_nb):
        colors = display.colors
        color = colors[tracker_nb % len(colors)]
        display.legends.append(mpatches.Patch(color=color, label=len(self.tracklet)))
        return colors[tracker_nb % len(colors)]

class SMC(Tracker): 

    def __init__(self, frame_nb, X0, state_variance, observation_variance, n_particles=20):
        super().__init__(frame_nb, X0, state_variance, observation_variance, count_threshold=count_threshold)

        self.particles = multivariate_normal(
            X0, cov=self.observation_covariance).rvs(n_particles)
        self.normalized_weights = np.ones(n_particles)/n_particles

    def update(self, observation, flow, frame_nb=None):
        if observation is not None: super().update(observation, frame_nb)
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
        cov = np.diag(self.state_covariance)
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
        color = super().fill_display(display, tracker_nb)
        display.ax.scatter(self.particles[:,0], self.particles[:,1], s=5, c=color)

class Kalman(Tracker): 

    def __init__(self, frame_nb, X0, state_variance, observation_variance, delta):
            super().__init__(frame_nb, X0, state_variance, observation_variance, delta)
            self.filter = KalmanFilter(initial_state_mean=X0, 
                                       initial_state_covariance=self.observation_covariance, 
                                       transition_matrices=np.eye(2), 
                                       transition_covariance=self.state_covariance,
                                       observation_matrices=np.eye(2),
                                       observation_covariance=self.observation_covariance)

            self.filtered_state_mean = X0
            self.filtered_state_covariance = self.observation_covariance

    def update(self, observation, flow, frame_nb=None):
        if observation is not None: super().update(observation, frame_nb)
        transition_offset = flow[int(self.filtered_state_mean[1]),int(self.filtered_state_mean[0]), :]

        self.filtered_state_mean, self.filtered_state_covariance = self.filter.filter_update(self.filtered_state_mean, 
                                                                                             self.filtered_state_covariance, 
                                                                                             observation=observation,
                                                                                             transition_offset=transition_offset)
        enabled=False if not in_frame(self.filtered_state_mean,flow.shape[:-1]) else True

        return enabled

    def predictive_distribution(self, flow):
        global legends
        transition_offset = flow[max(0, int(self.filtered_state_mean[1])), max(0, int(self.filtered_state_mean[0])), :]

        filtered_state_mean, filtered_state_covariance = self.filter.filter_update(self.filtered_state_mean, 
                                                                                             self.filtered_state_covariance, 
                                                                                             observation=None,
                                                                                             transition_offset=transition_offset)

        # distribution = multivariate_normal(filtered_state_mean, filtered_state_covariance)

        return distribution
    
    def fill_display(self, display, tracker_nb):
        yy, xx = np.mgrid[0:display.display_shape[1]:1, 0:display.display_shape[0]:1]
        pos = np.dstack((xx, yy))    
        distribution = multivariate_normal(self.filtered_state_mean, self.filtered_state_covariance)

        color = super().fill_display(display, tracker_nb)
        cs = display.ax.contour(distribution.pdf(pos), colors=color)
        display.ax.clabel(cs, inline=True, fontsize='large')
        display.ax.scatter(self.filtered_state_mean[0], self.filtered_state_mean[1], color=color, marker="x", s=100)

class DetectionFreeTracker:

    def __init__(self, heatmap0, jump_probability, state_variance, observation_variance, num_samples=500):
        self.state_covariance = np.diag(state_variance)/2
        # self.state_covariance = np.diag([1,1])
        self.observation_covariance = np.diag(observation_variance)

        self.num_samples = num_samples
        self.samples = []
        self.jump_probability = jump_probability
        self.discrete_distrib = MultivariateDiscrete(heatmap0.shape, heatmap0.ravel())
        self.samples.append(self.discrete_distrib.sample(num_samples))


    # def update(self, heatmap, flow):
    #     self.discrete_distrib.update_weights(heatmap.ravel())
    #     new_samples = np.zeros_like(self.samples[-1])
    #     draws = np.random.uniform(0, 1, self.num_samples)

    #     where_new = draws < self.jump_probability
    #     samples_from_heatmap = self.discrete_distrib.sample(where_new.sum())
    #     new_samples[where_new] = samples_from_heatmap

    #     indices_from_transition = np.argwhere(~where_new).ravel()
    #     for sample_id in indices_from_transition:
    #         sample = self.samples[-1][sample_id]
    #         mean = sample + flow[int(sample[1]),int(sample[0]), :]
    #         new_sample = multivariate_normal(mean, cov=self.state_covariance).rvs(1).astype(int)
    #         if in_frame(new_sample, heatmap.shape): new_samples[sample_id] = new_sample
    #         else: new_samples[sample_id] = self.discrete_distrib.sample(1)

    #     self.samples.append(new_samples)


    def update(self, heatmap, flow): 
        self.discrete_distrib.update_weights(heatmap.ravel())
        new_samples = np.zeros_like(self.samples[-1])
        for sample_id in range(self.num_samples):
            while True:
                if np.random.uniform(0, 1, 1) < self.jump_probability:
                    new_samples[sample_id] = self.discrete_distrib.sample(1)
                    break
                else: 
                    sample = self.samples[-1][sample_id]
                    mean = sample + flow[int(sample[1]),int(sample[0]), :]
                    candidate = multivariate_normal.rvs(mean=mean, cov=self.state_covariance, size=1).astype(int)
                    # candidate_x = norm.rvs(loc=mean[0], scale=self.variance_x, size=1)
                    # candidate_y = norm.rvs(loc=mean[1], scale = self.variance_y, size=1)
                    # candidate = np.array([candidate_x, candidate_y]).ravel()
                    if in_frame(candidate, heatmap.shape): 
                        new_samples[sample_id] = candidate
                        break
        self.samples.append(new_samples)

    

        

                    
           



        

trackers = {'Kalman':Kalman,
           'SMC':SMC,
           'DetectionFreeTracker':DetectionFreeTracker}
