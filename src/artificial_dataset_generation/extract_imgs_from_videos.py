import os
import cv2
import random
import sys


def extract_from_video(video_path, num_images, save_dir):
    # create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # get the total number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # randomly select x frame numbers
    frame_numbers = random.sample(range(num_frames), num_images)

    # iterate through the selected frames and extract the corresponding images
    for frame_number in frame_numbers:
        # set the frame number to read
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # read the frame
        ret, frame = cap.read()

        if ret:
            # save the frame as an image in the specified directory
            video_file = os.path.splitext(os.path.basename(video_path))[0]
            save_path = os.path.join(
                save_dir, f'{video_file}_image{frame_number}.jpg')
            cv2.imwrite(save_path, frame)

    # release the VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


def main(video_dir='./background_videos', save_dir='./extracted_background_images/', num_images=1):


    # create the save directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    # iterate over the video files in the directory
    for video_file in os.listdir(video_dir):
        # replace with the file extension of your video files
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            print(f'Extracting images from {video_path}')

            extract_from_video(video_path, num_images, save_dir)

if __name__ == "__main__":

    # replace with the number of images you want to extract
    num_images = 1
    # replace with your video file path
    video_dir = './background_videos'
    # replace with the path to the directory where you want to save the images
    save_dir = './extracted_background_images/'

    # Check if the parameters were provided
    if len(sys.argv) > 3:
        num_images = int(sys.argv[3])
    if len(sys.argv) > 2:
        save_dir = sys.argv[2]
    if len(sys.argv) > 1:
        video_dir = sys.argv[1]

    # Call the main function and pass the parameter
    main(video_dir, save_dir, num_images)

