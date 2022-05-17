def plot_image_and_bboxes(img, anns, ratio):

    """_summary_
    """
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(img)
    
    for ann in anns:
        
        [bbox_x, bbox_y, bbox_w, bbox_h] = (ratio*np.array(ann['bbox'])).astype(int)
        rect = patches.Rectangle((bbox_x, bbox_y), bbox_w, bbox_h, linewidth=2, edgecolor='r', facecolor="none")
        ax.add_patch(rect)
    
    plt.show()

# Creation of the bounding boxes on an image. 
# Takes as arguments : an image, annotations and a ratio. 
# Obtains the bbox coordinates via the ratio and anns for all the annotations of said image. 
# It prints out a 12 * 10 image with a bounding box. 

