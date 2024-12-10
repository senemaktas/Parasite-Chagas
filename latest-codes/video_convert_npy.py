import cv2
import numpy as np

dataset = np.load('my_video.npy')
# dataset.reshape((1, 3, 1080,1920))
print(dataset.shape)

# Swap the axes representing the number of frames and number of data samples.
# dataset = np.swapaxes(dataset, 0, 1)

# We'll pick out 1000 of the 10000 total examples and use those.
# dataset = dataset[:100, ...]

# Add a channel dimension since the images are grayscale.
# dataset = np.expand_dims(dataset, axis=-1)

# Split into train and validation sets using indexing to optimize memory.
indexes = np.arange(dataset.shape[0])
np.random.shuffle(indexes)
train_index = indexes[: int(0.9 * dataset.shape[0])]
val_index = indexes[int(0.9 * dataset.shape[0]):]
train_dataset = dataset[train_index]
val_dataset = dataset[val_index]

# Normalize the data to the 0-1 range.
train_dataset = train_dataset / 255
val_dataset = val_dataset / 255


# We'll define a helper function to shift the frames, where
# `x` is frames 0 to n - 1, and `y` is frames 1 to n.
def create_shifted_frames(data):
    x = data[:, 0: data.shape[1] - 1, :, :]
    y = data[:, 1: data.shape[1], :, :]
    return x, y


# Apply the processing function to the datasets.
x_train, y_train = create_shifted_frames(train_dataset)
x_val, y_val = create_shifted_frames(val_dataset)

# Inspect the dataset.
print("Training Dataset Shapes: " + str(x_train.shape) + ", " + str(y_train.shape))
print("Validation Dataset Shapes: " + str(x_val.shape) + ", " + str(y_val.shape))

# (215, 1080, 1920, 3)
# Training Dataset Shapes: (193, 1079, 1920, 3), (193, 1079, 1920, 3)
# Validation Dataset Shapes: (22, 1079, 1920, 3), (22, 1079, 1920, 3)

# (20, 10000, 64, 64)
# Training Dataset Shapes: (9000, 19, 64, 64), (9000, 19, 64, 64)
# Validation Dataset Shapes: (1000, 19, 64, 64), (1000, 19, 64, 64)

"""
# Video to Numpy Array
frames = []

path = "./chagas_out/chagas20_preprocess_out_cleaned_out_flow_dense.avi"
cap = cv2.VideoCapture(path)
ret = True
while ret:
    ret, img = cap.read()  # read one frame from the 'capture' object; img is (H, W, C)
    if ret:
        frames.append(img)
print(frames)
array_of_images = np.stack(frames, axis=0)  # dimensions (T, H, W, C)

np.save('my_video.npy', array_of_images)
"""
# Numpy Array to Video

# # let `video` be an array with dimensionality (T, H, W, C)
# num_frames, height, width, _ = video.shape
#
# filename = "/path/where/video/will/be/saved.mp4"
# codec_id = "mp4v" # ID for a video codec.
# fourcc = cv2.VideoWriter_fourcc(*codec_id)
# out = cv2.VideoWriter(filename, fourcc=fourcc, fps=20, frameSize=(width, height))
#
# for frame in np.split(video, num_frames, axis=0):
#     out.write(frame)
