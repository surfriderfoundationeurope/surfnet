import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import csv


class Colors:
    """
    A class for managing color palettes.
    """
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        """
        Get a color from the palette by index.

        Args:
            i (int): Index of the color.
            bgr (bool): Whether to return the color in BGR format (True) or RGB format (False).

        Returns:
            tuple: A tuple representing the color in RGB or BGR format.
        """
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """
        Convert a hexadecimal color code to an RGB tuple.

        Args:
            h (str): Hexadecimal color code.

        Returns:
            tuple: A tuple representing the color in RGB format.
        """
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def write_csv(counts, result_path):
    """
    Write class counts to a CSV file.

    Args:
        counts (dict): A dictionary containing class names as keys and counts as values.
        result_path (str): Path to the CSV file to write.
    """
    with open(result_path, 'w', newline='') as file:
        writer = csv.writer(file)
        header = ["class", "num_objects"]
        writer.writerow(header)
        for key in counts:
            writer.writerow([key, counts[key]])



def count_lines_with_number(files_train, result_path):
    """
    Count the number of objects for each class using the label files. The function counts lines with each class id.

    Args:
        folder_path (str): Path to the folder containing the label text files.
        result_path (str): Path to the CSV file to save the results.

    Returns:
        dict: A dictionary containing class names as keys and counts as values.
    """
    class_names = ["Tarp fragment", "Insulating material", "Bottle-shaped", "Can-shaped", "Drum",
                   "Other packaging", "Tire", "Fishing net / cord", "Easily namable", "Unclear", "Sheet", "Black Plastic"]
    # Initialize the counts dictionary
    counts = {i: 0 for i in class_names}
    for filename in files_train:
        filename = filename.replace(".jpg", ".txt")
        filename = filename.replace("images", "labels")
        if filename.endswith(".txt"):
            filepath = filename
            with open(filepath, "r") as file:
                for line in file:
                    first_character = line.split()[0]
                    if first_character.isdigit() and int(first_character) < 12:
                        counts[class_names[int(first_character)]] += 1

    write_csv(counts, result_path)
    return counts


def save_bar_chart(class_labels, result_path):
    """
    Generate and save a bar chart of class labels and counts.

    Args:
        class_labels (dict): A dictionary containing class names as keys and counts as values.
        result_path (str): Path to save the generated bar chart (PNG file).
    """

    class_names = list(class_labels.keys())
    label_counts = list(class_labels.values())

    nc = len(class_names)
    x_pos = np.arange(len(class_names))
    bar_plot = plt.bar(class_names, label_counts)
    for i in range(nc):
        bar_plot[i].set_color([x / 255 for x in colors(i)])

    plt.xlabel('Class Names')
    plt.ylabel('Instances')
    plt.title('Label Counts by Class')
    plt.xticks(x_pos, class_names, rotation=90)

    plt.subplots_adjust(bottom=0.35)
    # plt.grid(True)
    # plt.show()
    plt.savefig(result_path)
    plt.close()


def read_file_lines(file_path):
    """
    Read lines from a file and return them as a list of strings. This function is used to get the list of paths of the label text files from the train.txt and val.txt files.

    Args:
        file_path (str): Path to the file to read.

    Returns:
        list: A list of strings representing the lines of the file.
    """

    lines = []
    with open(file_path, 'r') as file:
        for line in file:
            lines.append(line.strip())
    return lines


def main(train_path, results_path, val_path, csv_labels_file_name, png_labels_file_name):
    """
    Main function to count and visualize class labels.

    Args:
        train_path (str): Path to the train.txt file.
        results_path (str): Path to the folder where the CSV and PNG files will be saved.
        val_path (str): Path to the val.txt file (optional).
        csv_labels_file_name (str): Name of the CSV file to be generated.
        png_labels_file_name (str): Name of the PNG file (bar chart) to be generated.
    """

    # get the list of files in the train.txt file
    files_train = read_file_lines(train_path)
    if val_path != "" and val_path is not None:
        files_val = read_file_lines(val_path)
        files_train += files_val

    labels_csv = os.path.join(results_path, csv_labels_file_name)
    labels_png = os.path.join(results_path, png_labels_file_name)
    class_labels = count_lines_with_number(files_train, labels_csv)

    # draw a line chart with each class name and number of labels
    save_bar_chart(class_labels, labels_png)


if __name__ == "__main__":

    # Define default paths
    default_csv_labels_file_name = 'labels.csv'
    default_png_labels_file_name = 'labels.png'
    val_path = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str,
                        help="Path to the train.txt file")
    parser.add_argument("--val_path", type=str,
                        help="Path to the val.txt file (optional) ")
    parser.add_argument("--results_path", type=str,
                        help="Path to the folder where the graph and csv will be saved")
    parser.add_argument("--csv_labels_file_name", type=str,
                        default=default_csv_labels_file_name, help="Name of the csv labels file that will be generated")
    parser.add_argument("--png_labels_file_name", type=str,
                        default=default_png_labels_file_name, help="Name of the png labels file that will be generated")
    args = parser.parse_args()

    train_path = args.train_path
    val_path = args.val_path
    results_path = args.results_path
    csv_labels_file_name = args.csv_labels_file_name
    png_labels_file_name = args.png_labels_file_name

    # Call the main function and pass the parameters
    main(train_path, results_path, val_path,
         csv_labels_file_name, png_labels_file_name)
