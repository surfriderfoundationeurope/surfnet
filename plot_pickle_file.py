import pickle 
import matplotlib.pyplot as plt 

def plot_pickle_file(file_name):

    with open(file_name,'rb') as f :
        fig, axes = pickle.load(f)
    # plt.savefig(file_name.strip('_axes.pickle'))
    print(file_name.strip('_axes.pickle'))
    plt.show()
    plt.close()
    del fig
    del axes



# plot_pickle_file('Evaluation base_axes.pickle')
plot_pickle_file('Evaluation base nms_axes.pickle')
# plot_pickle_file('Evaluation extension_axes.pickle')
plot_pickle_file('Evaluation extension nms_axes.pickle')


