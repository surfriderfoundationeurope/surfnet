import pandas as pd 
import matplotlib.pyplot as plt


def annotations_cat (df_bboxes):

    """Function returning a plot and table with the proportion of trash by category. 

    Args: 
        df_bboxes(csv file): Csv file with the bounding boxes description, coordinates, ID and linked image.

    Returns: 
        plot(plot): Plot of the count of the categories x-axis : label category, y-axis count.
        table(dataframe): ID, percentage and count of the different trash categories.

    """

    categories = ['N/A', 'Sheet / tarp / plastic bag / fragment', 'Insulating material','Drum', 'Bottle-shaped', 
                  'Can-shaped', 'Other packaging', 'Tire', 'Fishing net / cord', 'Easily namable', 'Unclear']
                  # trash categories 

    cat_percentages = round((df_bboxes['id_ref_trash_type_fk'].value_counts()/len(df_bboxes)*100),2).to_frame()
    # percentages of each trash category in the dataset

    table = pd.DataFrame(categories)      # creation of data frame of the categories 
    table["Percentage"] = cat_percentages # adding the percentages to the df
    table["ID"] = list(range(0,11))       # adding the ID to the df 
    table["Count"] = df_bboxes['id_ref_trash_type_fk'].value_counts().to_frame() # adding the count of the labels to the df 
    table.rename(columns = {0 : 'Category'}, inplace = True)                     # rename the category column 
    new_table = table.set_index('ID').drop([0])                                  # final df, dropping excess info 

    fig, ax = plt.subplots(figsize=(15,7)) # creation 
    dfg = df_bboxes.groupby(['id_ref_trash_type_fk']).size()
    plot = dfg.plot(kind='bar', title='Trash Size by ID', ylabel='Size', xlabel='Trash ID', figsize=(6, 5))
    ax.legend(title='Categories',loc='center left', bbox_to_anchor=(1, 0.5))

    handles = [plt.Rectangle((0,0),1,1) for label in categories]
    plt.legend(handles, categories[1:11], bbox_to_anchor=(1.02, .75), title='Categories')
    
    return(print(plot), print(new_table))




def img_context (df_images):

    """ Function creating a plot with the number of pictures depending on their context. 

    Args:
        df_images (data frame): Data frame with the information of the images/pictures used. 

    Returns:
        plot (plot): Plot of the context per label category. 
        table (data frame): Percentages and count of the images by context. 
    """

    context_percent = round((df_images['context'].value_counts()/len(df_images)*100),2)

    table = pd.DataFrame(context_percent)
    table["Count"] = df_images['context'].value_counts()
    table.rename(columns = {0 : 'Context', 'context' : 'Percentages'}, inplace = True)

    fig, ax = plt.subplots(figsize=(15,7))
    dfg = df_images.groupby(['context']).size()
    plot = dfg.plot(kind='bar', title='Size by Context', ylabel='Size',
         xlabel='Context', figsize=(6, 5))
    
    return(print(table, plot))