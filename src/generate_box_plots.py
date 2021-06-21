import pandas as pd
import matplotlib.pyplot as plt

tracker_names = ['Our_tracker_FairMOT_detections_clean_0','FairMOT_clean_0','Our_tracker_FairMOT_detections_clean_1','FairMOT_clean_1','Our_detections_our_tracker']
results = [pd.read_csv('/home/infres/chagneux/repos/TrackEval/data/trackers/surfrider_segments_T1/surfrider-test/{}/pedestrian_detailed.csv'.format(tracker_name),',') \
    for tracker_name in tracker_names]


count_errors_relative =  pd.DataFrame({tracker_name: pd.Series((tracker_results['IDs'][:-1]-tracker_results['GT_IDs'][:-1]))/tracker_results['GT_IDs'][:-1] for tracker_name,tracker_results in zip(tracker_names,results)})
count_errors =  pd.DataFrame({tracker_name: pd.Series((tracker_results['IDs'][:-1]-tracker_results['GT_IDs'][:-1])) for tracker_name,tracker_results in zip(tracker_names,results)})
 
# print(count_errors)
# print(count_errors_relative)
count_errors.columns = tracker_names
count_errors_relative.boxplot()
plt.suptitle('Box plot on 19 consecutive short sequences from T1')
plt.ylabel(r'$\frac{\hat{N}-N}{N}$')
plt.show()





