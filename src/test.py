# from common.utils import open_pickle

from tracking.utils import discretized_gaussian
import numpy as np 
import matplotlib.pyplot as plt 

# pickle_filename = 'data/external_detections/CenterTrack/T1_1080_px_converted.pickle'
# data = open_pickle(pickle_filename)

# output_file = open(pickle_filename.replace('.pickle','.txt'),'w')

# results = []
# for frame_nb, detections_for_frame in data.items(): 
#     if len(detections_for_frame):
#         for detection in detections_for_frame:
#             results.append((frame_nb, detection['tracking_id'], detection['ct'][0], detection['ct'][1]))

# ratio_x = 1920 / 960

# ratio_y = 1080 / 544

# for result in results:
#     output_file.write('{},{},{},{},{},{},{},{},{},{}\n'.format(result[0]+1,
#                                                             result[1],
#                                                             ratio_x * result[2],
#                                                             ratio_y * result[3],
#                                                             -1,
#                                                             -1,
#                                                             1,
#                                                             -1,
#                                                             -1,
#                                                             -1))

result = discretized_gaussian(mean=np.array([240/2,136/2]), var=np.array([2,2]), shape=(136,240))

# plt.imshow(result, cmap='gray')
# plt.show()
test = 0 
