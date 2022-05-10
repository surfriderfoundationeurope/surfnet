def path_existance(img_ids, data_dir:file=images2labels) :

    """_summary_

    Args:
        img_ids (_type_): _description_
        data_dir (file, optional): _description_. Defaults to images2labels.

    Returns:
        my_df (data frame): 
    """

    for img_id in img_ids:
        image_infos = coco.loadImgs(ids=[img_id])[0]

        if os.path.exists(os.path.join(data_dir, image_infos['file_name'])):
            
            # concatenate the data directory path and the files from the coco transformation ; if it exists, we compute the if loop.

            date_creation  = df_images.loc[df_images["filename"] == image_infos["file_name"]]["createdon"].values[0]
            view           = df_images.loc[df_images["filename"] == image_infos["file_name"]]["view"].values[0]
            image_quality  = df_images.loc[df_images["filename"] == image_infos["file_name"]]["image_quality"].values[0]
            context        = df_images.loc[df_images["filename"] == image_infos["file_name"]]["context"].values[0]

            date_time_obj = datetime.datetime.strptime(date_creation, '%Y-%m-%d %H:%M:%S')

            old_filenames.append(image_infos["file_name"])
            dates.append(date_time_obj)
            views.append(view)
            images_quality.append(image_quality)
            contexts.append(context)

            # in the loop we put the info of the image corresponding to the date, the type of view, the quality and the context to the empty lists created in previous function.

            image = Image.open(os.path.join(data_dir,image_infos['file_name']))
            try:
                for orientation in ExifTags.TAGS.keys():
                    if ExifTags.TAGS[orientation]=='Orientation':
                        break
                exif = image._getexif()
                if exif is not None:
                    if exif[orientation] == 3:
                        image=image.rotate(180, expand=True)
                    elif exif[orientation] == 6:
                        image=image.rotate(270, expand=True)
                    elif exif[orientation] == 8:
                        image=image.rotate(90, expand=True)

            except (AttributeError, KeyError, IndexError):
                # cases: image don't have getexif
                pass

            image    = np.array(image) #cv2.cvtColor(np.array(image.convert('RGB')),  cv2.COLOR_RGB2BGR)
            ann_ids  = coco.getAnnIds(imgIds=[img_id])
            anns     = coco.loadAnns(ids=ann_ids)
            h, w     = image.shape[:-1]
            target_h = 1080
            ratio    = target_h/h
            target_w = int(ratio*w) 
            image    = cv2.resize(image,(target_w,target_h))
            h, w     = image.shape[:-1]
            yolo_annot = []
            for ann in anns:
                cat = ann['category_id'] - 1
                [bbox_x, bbox_y, bbox_w, bbox_h] = (ratio*np.array(ann['bbox'])).astype(int)
                bbox = np.array([bbox_x, bbox_y, bbox_w, bbox_h])
                yolo_bbox = coco2yolo(bbox, target_h, target_w)
                yolo_str  = str(cat) + " " + " ".join(yolo_bbox.astype(str))
                yolo_annot.append(yolo_str)
            
            basename  = os.path.splitext(image_infos['file_name'])[0]
            file_name = str(image_infos['id']) + "-" + basename

            img_file_name   = os.path.join("./images", file_name) + ".jpg"
            label_file_name = os.path.join("./labels", file_name) + ".txt"
            
            # Save Label
            with open(label_file_name, 'w') as f:
                f.write('\n'.join(yolo_annot))

            img_to_save = Image.fromarray(image)
            # Save image
            img_to_save.save(img_file_name)

            all_bboxes.append(yolo_annot)
            new_filenames.append(img_file_name)
            new_labelnames.append(label_file_name)
            all_images.append(img_to_save)
            
    my_list = list(zip(old_filenames, dates, views, images_quality, contexts, new_filenames, new_labelnames, all_images, all_bboxes))
    my_df   = pd.DataFrame(my_list, columns=['old_path', 'date', 'view', 'quality','context', 'img_name', 'label_name', 'img', 'bboxes'])

    return(my_df)