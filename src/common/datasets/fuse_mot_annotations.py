import numpy as np 

# mot_results1  = np.loadtxt('gt1.txt', delimiter=',')
mot_results2  = np.loadtxt('gt2.txt', delimiter=',')

# num_frames_1 = 6474

# mot_results2[:,0]+=num_frames_1
# mot_results2[:,1]+=mot_results1[:,1].max()

# mot_results = np.concatenate([mot_results1,mot_results2])

with open('gt.txt','w') as out_file:
        for mot_result in mot_results2:
            frame_id = int(mot_result[0])
            track_id = int(mot_result[1])
            left, top, width, height = mot_result[2:6]
            center_x = left+width/2
            center_y = top+height/2
            if (frame_id - 1) % 2 == 0: 
                out_file.write('{},{},{},{},{},{},{},{},{},{}\n'.format(int((frame_id - 1) // 2 + 1),
                                                                        int(track_id),
                                                                        center_x,
                                                                        center_y,
                                                                        -1,
                                                                        -1,
                                                                        1,
                                                                        -1,
                                                                        -1,
                                                                        -1))


