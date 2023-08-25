import os
import cv2
import random
import argparse


def extract_from_video(video_path, num_images, save_dir):
    """Extracts a specific number of random frames from a video and saves them as images.

    Args:
        video_path (str): Path to the video file.
        num_images (int): Number of random frames to extract.
        save_dir (str): Directory to save the extracted images.
    """

    # create a VideoCapture object
    num_images = int(num_images)
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


def main(video_dir, save_dir, num_images):
    """Main function to extract images from multiple videos.

    Args:
        video_dir (str): Directory containing the video files.
        save_dir (str): Directory to save the extracted images.
        num_images (str): Number of images to extract from each video.
    """

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
    parser = argparse.ArgumentParser()

    # Define default paths
    default_num_images = 1
    default_video_dir_path = '../background_videos'
    default_save_dir_path = '../extracted_background_images/'

    # Add arguments
    parser.add_argument("--num_images", type=str,
                        default=default_num_images, help="The number of images extracted from each video")
    parser.add_argument("--video_dir_path", type=str,
                        default=default_video_dir_path, help="Path to the folder containing the videos")
    parser.add_argument("--save_dir_path", type=str,
                        default=default_save_dir_path, help="Path to the folder where to save the extracted images")

    args = parser.parse_args()

    num_images = args.num_images
    video_dir = args.video_dir_path
    save_dir = args.save_dir_path

    # Call the main function and pass the parameter
    main(video_dir, save_dir, num_images)
