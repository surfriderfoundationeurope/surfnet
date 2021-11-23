from torch.utils.data.dataloader import DataLoader
from detection.transforms import TransformFrames
from tools.misc import load_model
from tools.video_readers import IterableFrameReader, TorchIterableFromReader
from torch.profiler import profile, record_function, ProfilerActivity, schedule
from detection.detect import nms
import torch 
import numpy as np 
import time

# def trace_handler(prof):
#     print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

weights_path = {'res_18':'models/res18_pretrained.pth',
                'dla_34':'models/dla_34_pretrained.pth',
                'mobilenetv3small':'models/mobilenet_v3_pretrained.pth'}

video_filename = 'data/validation_videos/T1/T1_1080_px_converted.mp4'

output_shape = (960,544)
batch_size = 8
device = 'cuda'
arch = 'res_18'
skip_frames = 0
threshold = 0.3 

if device == 'cpu': batch_size = 1
device = torch.device(device)
reader = IterableFrameReader(video_filename, skip_frames=skip_frames, output_shape=output_shape)
dataset = TorchIterableFromReader(reader, TransformFrames())
loader = DataLoader(dataset=dataset, batch_size=batch_size)
model = load_model(arch, weights_path[arch], device)


# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

timings = []

with torch.no_grad():
    for iter_nb, frames in enumerate(loader): 

        starter = time.time() 
        # starter.record()
        frames = frames.to(device)
        model_output = model(frames)
        # ender.record()
        # batch_peaks = nms(torch.sigmoid(model_output[-1]['hm'])).gt(threshold).squeeze()
        # detections = [torch.nonzero(peaks).cpu().numpy()[:,::-1] for peaks in batch_peaks]
        # torch.cuda.synchronize()
        timings.append(time.time() - starter)
        # timings.append(starter.elapsed_time(ender))

        if iter_nb*batch_size > 100: break

print(f'Mean inference speed: {batch_size/np.mean(timings):.2f} fps')


# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
