fig, ax = plt.subplots(figsize=(15,7))
dfg = df_bboxes.groupby(['id_ref_trash_type_fk']).size()
dfg.plot(kind='bar', title='Trash Size by ID', ylabel='Size',
         xlabel='Trash ID', figsize=(6, 5))

cat_percentages = df_bboxes['id_ref_trash_type_fk'].value_counts()/len(df_bboxes)*10
print(cat_percentages)
    """
    Barplot describing the amount of objects labeled per category, and their representation percentages.
    1: 'Sheet / tarp / plastic bag / fragment',
    2: 'Insulating material',
    3: 'Bottle-shaped',
    4: 'Can-shaped',
    5: 'Drum',
    6: 'Other packaging',
    7: 'Tire',
    8: 'Fishing net / cord',
    9: 'Easily namable',
    10: 'Unclear'
    """