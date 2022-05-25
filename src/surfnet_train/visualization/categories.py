


def annotations_cat (df_bboxes):

    """_summary_
    """

    cat_percentages = round((df_bboxes['id_ref_trash_type_fk'].value_counts()/len(df_bboxes)*100),2).to_frame()
    
    categories = ['N/A', 'Sheet / tarp / plastic bag / fragment', 'Insulating material','Drum', 'Bottle-shaped', 
                  'Can-shaped', 'Other packaging', 'Tire', 'Fishing net / cord', 'Easily namable', 'Unclear']

    table = pd.DataFrame(categories)
    table["Percentage"] = cat_percentages
    table["ID"] = list(range(0,11))
    table["Count"] = df_bboxes['id_ref_trash_type_fk'].value_counts().to_frame()
    table.drop([0])
    table.rename(columns = {0 : 'Category'}, inplace = True)
    new_table = table.set_index('ID').drop([0])

    fig, ax = plt.subplots(figsize=(15,7))
    dfg = df_bboxes.groupby(['id_ref_trash_type_fk']).size()
    plot = dfg.plot(kind='bar', title='Trash Size by ID', ylabel='Size', xlabel='Trash ID', figsize=(6, 5))
    ax.legend(title='Categories',loc='center left', bbox_to_anchor=(1, 0.5))

    handles = [plt.Rectangle((0,0),1,1) for label in categories]
    plt.legend(handles, categories[1:11], bbox_to_anchor=(1.02, .75), title='Categories')
    
    return(print(plot), print(new_table))