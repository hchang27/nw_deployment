from ml_logger import logger

log_dict = "/alanyu/scratch/2024/06-01/225616/log_dict.pkl"

(data,) = logger.load_pkl(log_dict)

print(data.keys())

len(data["vision"])

images = []
for frame in data["vision"]:
    images.append(frame[0][-1].permute(1, 2, 0).cpu().numpy() + 0.5)

logger.save_video(images, "video_2.mp4", fps=50)
print(logger.get_dash_url())
