# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import cv2
import random
import tqdm


class BackgroundMatting(object):
    """
    Produce a mask by masking the given color. This is a simple strategy
    but effective for many games.
    """

    def __init__(self, color):
        """
        Args:
            color: a (r, g, b) tuple or single value for grayscale
        """
        self._color = color

    def get_mask(self, img):
        return img == self._color


class ImageSource(object):
    """
    Source of natural images to be added to a simulated environment.
    """

    def get_image(self):
        """
        Returns:
            an RGB image of [h, w, 3] with a fixed shape.
        """
        pass

    def reset(self):
        """Called when an episode ends."""
        pass


class FixedColorSource(ImageSource):
    def __init__(self, shape, color):
        """
        Args:
            shape: [h, w]
            color: a 3-tuple
        """
        self.arr = np.zeros((shape[0], shape[1], 3))
        self.arr[:, :] = color

    def get_image(self):
        return self.arr


class RandomColorSource(ImageSource):
    def __init__(self, shape):
        """
        Args:
            shape: [h, w]
        """
        self.shape = shape
        self.arr = None
        self.reset()

    def reset(self):
        self._color = np.random.randint(0, 256, size=(3,))
        self.arr = np.zeros((self.shape[0], self.shape[1], 3))
        self.arr[:, :] = self._color

    def get_image(self):
        return self.arr


class NoiseSource(ImageSource):
    def __init__(self, shape, strength=255):
        """
        Args:
            shape: [h, w]
            strength (int): the strength of noise, in range [0, 255]
        """
        self.shape = shape
        self.strength = strength

    def get_image(self):
        return np.random.randn(self.shape[0], self.shape[1], 3) * self.strength


class RandomImageSource(ImageSource):
    def __init__(self, shape, filelist, total_frames=None, grayscale=False):
        """
        Args:
            shape: [h, w]
            filelist: a list of image files
        """
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = filelist
        self.build_arr()
        self.current_idx = 0
        self.reset()

    def build_arr(self):
        self.total_frames = (
            self.total_frames if self.total_frames else len(self.filelist)
        )
        self.arr = np.zeros(
            (self.total_frames, self.shape[0], self.shape[1])
            + ((3,) if not self.grayscale else (1,))
        )
        for i in range(self.total_frames):
            # if i % len(self.filelist) == 0: random.shuffle(self.filelist)
            fname = self.filelist[i % len(self.filelist)]
            if self.grayscale:
                im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)[..., None]
            else:
                im = cv2.imread(fname, cv2.IMREAD_COLOR)
            self.arr[i] = cv2.resize(
                im, (self.shape[1], self.shape[0])
            )  ## THIS IS NOT A BUG! cv2 uses (width, height)

    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)

    def get_image(self):
        return self.arr[self._loc]


class RandomVideoSource(ImageSource):
    def __init__(self, shape, filelist, total_frames, grayscale=False):
        """
        Args:
            shape: [h, w]
            filelist: a list of video files
        """
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = filelist
        self.build_arr()
        self.current_idx = 0
        self.reset()

    def build_arr(self):
        self.arr = np.zeros(
            (self.total_frames, self.shape[0], self.shape[1])
            + ((3,) if not self.grayscale else (1,))
        )
        total_frame_i = 0
        file_i = 0
        with tqdm.tqdm(
            total=self.total_frames, desc="Loading videos for natural"
        ) as pbar:
            while total_frame_i < self.total_frames:
                if file_i % len(self.filelist) == 0:
                    random.shuffle(self.filelist)
                file_i += 1

                filename = self.filelist[file_i % len(self.filelist)]
                frames = self._get_frames(filename)
                for frame in frames:
                    if total_frame_i >= self.total_frames:
                        break
                    if self.grayscale:
                        self.arr[total_frame_i] = cv2.resize(
                            frame, (self.shape[1], self.shape[0])
                        )[
                            ..., None
                        ]  ## THIS IS NOT A BUG! cv2 uses (width, height)
                    else:
                        self.arr[total_frame_i] = cv2.resize(
                            frame, (self.shape[1], self.shape[0])
                        )
                    pbar.update(1)
                    total_frame_i += 1

    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)

    def get_image(self):
        img = self.arr[self._loc % self.total_frames]
        self._loc += 1
        return img

    def _get_frames(self, filename):
        cap = cv2.VideoCapture(filename)

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if self.grayscale:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frames.append(gray_frame)
        cap.release()

        return frames
