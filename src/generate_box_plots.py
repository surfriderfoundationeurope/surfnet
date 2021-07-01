import pandas as pd
import matplotlib.pyplot as plt

tracker_original_names = ['fairmot_detections_our_tracker','fairmot_count_thres_2','SORT_keep_all']
results = [pd.read_csv('/home/infres/chagneux/repos/TrackEval/data/trackers/surfrider_T1_segmented/surfrider-test/{}/pedestrian_detailed.csv'.format(tracker_name),',') \
    for tracker_name in tracker_original_names]

tracker_names = ['Ours','FairMOT','SORT']


count_errors_relative = pd.DataFrame({tracker_name: pd.Series((tracker_results['IDs'][:-1]-tracker_results['GT_IDs'][:-1]))/tracker_results['GT_IDs'][:-1] for tracker_name,tracker_results in zip(tracker_names,results)})
count_errors = pd.DataFrame({tracker_name: pd.Series((tracker_results['IDs'][:-1]-tracker_results['GT_IDs'][:-1])) for tracker_name,tracker_results in zip(tracker_names,results)})
 
print(count_errors_relative.mean())
print(count_errors_relative.std())
# print(count_errors_relative)
count_errors.columns = tracker_names
count_errors_relative.boxplot()
# plt.suptitle('Box plot on 17 independant short sequences from T1')
plt.ylabel(r'$\frac{\hat{N}-N}{N}$')
plt.tight_layout()
plt.savefig('boxplot',format='pdf')
plt.show()



