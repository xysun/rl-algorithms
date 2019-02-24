import numpy as np
import scipy.ndimage


def dqn_preprocess(frames, frames_to_stack=4, cropped_size=128.):
    '''
    preprocess following dqn nature paper
    1. remove flickering: max(pixel) over previous frame
    2. Convert RGB to greyscale
    3. rescale to 84x84
    4. do 1-4 for 4 frames and stack them
    :param frames: last 5 frames of shape (210, 160, 3), we need 5 to perform the remove flickering step
    :return: (84,84,4)
    '''
    RGB = 256
    assert len(frames) == frames_to_stack + 1

    processed_frames = []
    for i in range(1, frames_to_stack + 1):
        current_frame = frames[i]
        prev_frame = frames[i - 1]
        # step 1
        frame = np.maximum(current_frame, prev_frame)
        # step 2
        # first normalise
        assert np.amax(frame) <= RGB
        frame = np.divide(frame, RGB)
        frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])  # (210, 160)
        # step 3
        frame = scipy.ndimage.interpolation.zoom(frame, zoom=np.divide(cropped_size, frame.shape))

        processed_frames.append(frame)

    return np.dstack(processed_frames)
