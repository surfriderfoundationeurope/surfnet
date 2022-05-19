def image_orientation (image):

    """ Function which gives the images that have a specified orientation the same orientation.
        If the image does not have an orientation, the image is not altered. 

     Args:
            image (image object): Image that is in the path data_directory as well as in the instance json files. 

    Returns: Image, with the propeer orinetation. 
            _type_: image
    """

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

    return (image)