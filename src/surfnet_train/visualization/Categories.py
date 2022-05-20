import matplotlib.pyplot as plt

def annotations_cat (df_bboxes):
    
    """ Function creating a plot with the number of annotations per label category. 

    Args: 
        df_bboxes (): Data frame with the bounding boxes information. 
    
    Returns:
        plot (): Plot of the annotations per label category. 
        cat_percentages (): Percentages of the annotations per label category. 
    """
    categories = ['1 : Sheet / tarp / plastic bag / fragment', '2 : Insulating material','3 : Drum', '4 : Bottle-shaped', 
                  '5 : Can-shaped', '6 : Other packaging', '7 : Tire', '8 : Fishing net / cord', '9 : Easily namable', '10 : Unclear']

    fig, ax = plt.subplots(figsize=(15,7))
    dfg = df_bboxes.groupby(['id_ref_trash_type_fk']).size()
    plot = dfg.plot(kind='bar', title='Trash Size by ID', ylabel='Size', xlabel='Trash ID', figsize=(6, 5))
    
    cat_percentages = df_bboxes['id_ref_trash_type_fk'].value_counts()/len(df_bboxes)*10

    handles = [plt.Rectangle((0,0),1,1) for label in categories]
    plt.legend(handles, categories, bbox_to_anchor=(1.02, .75), title='Categories')

    return(plot, cat_percentages)




def img_context (df_images):

    """ Function creating a plot with the number of pictures depending on their context. 

    Args:
        df_images (): Data frame with the information of the images/pictures used. 

    Returns:
        plot (): Plot of the context per label category. 
    """

    fig, ax = plt.subplots(figsize=(15,7))
    dfg = df_images.groupby(['context']).size()
    plot = dfg.plot(kind='bar', title='Context Size by Context', ylabel='Size',
         xlabel='Context', figsize=(6, 5))
    
    return(plot)




def img_view (df_images):

    """ Function creating a plot with the number of pictures grouped by their view.

        Args:
            df_images (): Data frame with the information of the images/pictures used. 
        
        Returns:
            plot (): Plot of the view per label category. 
    """

    fig, ax = plt.subplots(figsize=(15,7))
    dfg = df_images.groupby(['view']).size()
    plot = dfg.plot(kind='bar', title='View Size by View', ylabel='Size',
         xlabel='View', figsize=(6, 5))

    return(plot)



def img_quality (df_images):

    """Function creating a plot with the number of pictures grouped by their quality.

    Args:
            df_images (): Data frame with the information of the images/pictures used. 
        
        Returns:
            plot (): Plot of the quality per label category. 
    """

    fig, ax = plt.subplots(figsize=(15,7))
    dfg = df_images.groupby(['image_quality']).size()
    plot = dfg.plot(kind='bar', title='Image Quality Size by Image Quality', ylabel='Size',
        xlabel='Image Quality', figsize=(6, 5))
    
    return(plot)