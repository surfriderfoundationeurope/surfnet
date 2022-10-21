# from plasticorigins.tools.video_readers import SimpleVideoReader
# from plasticorigins.serving.inference import config_track


# def test_track_video_simple():
#     config_track.video_path = "tests/ressources/validation_videos/T1_trim.mp4"

#     reader = SimpleVideoReader(
#         video_filename=config_track.video_path,
#         skip_frames=config_track.skip_frames,
#     )

#     assert reader.read()[0]
#     assert reader.read()[1].shape == (360, 640, 3)
#     #assert reader.read()[-1] == 1
