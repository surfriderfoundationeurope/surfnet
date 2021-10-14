import cProfile, pstats, io
from pstats import SortKey

def profile_dataset(dataset):

    pr = cProfile.Profile()

    pr.enable()

    next(iter(dataset))    

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

heatmaps_folder = "data/extracted_heatmaps/"
annotations_dir = 'data/generated_videos/'

video_dataset = SurfnetDatasetFlow(annotations_dir=annotations_dir, heatmaps_folder=heatmaps_folder, split='val')

test = next(iter(video_dataset))
