import pickle 

with open("data/detector_results/real_val/CenterTrack/real/T1_1080_px_converted.pickle",'rb') as f: 

    detections = pickle.load(f)
    test = 0 