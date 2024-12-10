# from motiongram_flow import Flow  # created this filee from original one
import os
import cv2
import numpy as np
import math
# from motiongram_utils import MgProgressbar, generate_outfilename


class DenseOpticalFlow:

    def __init__(self, video_path):
        self.video_path = video_path

    def get_dense_flow(self):
        saved_video_path = Flow(filename=self.video_path, color=False,
                                has_audio=False).dense(pyr_scale=0.2, levels=3, winsize=5, iterations=3, poly_n=7,
                                                       poly_sigma=1.5, timestep=1, move_step=1,
                                                       skip_empty=True, velocity=False)
        return saved_video_path


class Flow:

    def __init__(self, filename, color, has_audio):
        """
        Initializes the Flow class.

        Args:
            parent (MgVideo): the parent MgVideo.
            filename (str): Path to the input video file. Passed by parent MgVideo.
            color (bool): Set class methods in color or grayscale mode. Passed by parent MgVideo.
            has_audio (bool): Indicates whether source video file has an audio track. Passed by parent MgVideo.
        """
        # self.parent = weakref.ref(parent)
        self.filename = filename
        self.color = color
        self.has_audio = has_audio

    @staticmethod
    def get_acceleration(velocity, fps):
        acceleration = np.zeros(len(velocity))
        velocity = np.abs(velocity)
        for i in range(len(acceleration) - 1):
            acceleration[i] = ((velocity[i + 1] + velocity[i]) - velocity[i]) / (1 / fps)
        return acceleration[:-1]

    def get_velocity(self, flow, sum_flow_pixels, flow_shape, distance_meters, timestep_seconds, move_step,
                     angle_of_view):
        pixel_count = (flow.shape[0] * flow.shape[1]) / move_step ** 2
        average_velocity_pixels_per_second = (sum_flow_pixels / pixel_count / timestep_seconds)

        return (self.velocity_meters_per_second(average_velocity_pixels_per_second, flow_shape, distance_meters,
                                                angle_of_view)
                if angle_of_view and distance_meters else average_velocity_pixels_per_second)

    @staticmethod
    def velocity_meters_per_second(velocity_pixels_per_second, flow_shape, distance_meters, angle_of_view):
        distance_pixels = ((flow_shape / 2) / math.tan(angle_of_view / 2))
        pixels_per_meter = distance_pixels / distance_meters
        return velocity_pixels_per_second / pixels_per_meter

    def dense(self, filename=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5,
              poly_sigma=1.2, flags=0, velocity=False, distance=None, timestep=1, move_step=1,
              angle_of_view=0, scaledown=1, skip_empty=False, target_name=None, overwrite=False):
        """
        Renders a dense optical flow video of the input video file using `cv2.calcOpticalFlowFarneback()`.
        The description of the matching parameters are taken from the cv2 documentation.

        Args:
            filename (str, optional): Path to the input video file. If None the video file of the MgVideo is used. Defaults to None.
            pyr_scale (float, optional): Specifies the image scale (<1) to build pyramids for each image. `pyr_scale=0.5` means a classical pyramid, where each next layer is twice smaller than the previous one. Defaults to 0.5.
            levels (int, optional): The number of pyramid layers including the initial image. `levels=1` means that no extra layers are created and only the original images are used. Defaults to 3.
            winsize (int, optional): The averaging window size. Larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field. Defaults to 15.
            iterations (int, optional): The number of iterations the algorithm does at each pyramid level. Defaults to 3.
            poly_n (int, optional): The size of the pixel neighborhood used to find polynomial expansion in each pixel. Larger values mean that the image will be approximated with smoother surfaces, yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7. Defaults to 5.
            poly_sigma (float, optional): The standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion. For `poly_n=5`, you can set `poly_sigma=1.1`, for `poly_n=7`, a good value would be `poly_sigma=1.5`. Defaults to 1.2.
            flags (int, optional): Operation flags that can be a combination of the following: - **OPTFLOW_USE_INITIAL_FLOW** uses the input flow as an initial flow approximation. - **OPTFLOW_FARNEBACK_GAUSSIAN** uses the Gaussian \\f$\\texttt{winsize}\\times\\texttt{winsize}\\f$ filter instead of a box filter of the same size for optical flow estimation. Usually, this option gives z more accurate flow than with a box filter, at the cost of lower speed. Normally, `winsize` for a Gaussian window should be set to a larger value to achieve the same level of robustness. Defaults to 0.
            velocity (bool, optional): Whether to compute optical flow velocity or not. Defaults to False.
            distance (int, optional): Distance in meters to image (focal length) for returning flow in meters per second. Defaults to None.
            timestep (int, optional): Time step in seconds for returning flow in meters per second. Defaults to 1.
            move_step (int, optional): step size in pixels for sampling the flow image. Defaults to 1.
            angle_of_view (int, optional): angle of view of camera, for reporting flow in meters per second. Defaults to 0.
            scaledown (int, optional): factor to scaledown frame size of the video. Defaults to 1.
            skip_empty (bool, optional): If True, repeats previous frame in the output when encounters an empty frame. Defaults to False.
            target_name (str, optional): Target output name for the video. Defaults to None (which assumes that the input filename with the suffix "_flow_dense" should be used).
            overwrite (bool, optional): Whether to allow overwriting existing files or to automatically increment target filenames to avoid overwriting. Defaults to False.
        """

        if filename is None:
            filename = self.filename

        of, fex = os.path.splitext(filename)

        vidcap = cv2.VideoCapture(filename)

        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

        size = (int(width / scaledown), int(height / scaledown))

        pb = MgProgressbar(total=length, prefix='Rendering dense optical flow video:')

        if target_name is None:
            target_name = of + '_flow_dense' + fex
        if not overwrite:
            target_name = generate_outfilename(target_name)

        fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # MJPG
        out = cv2.VideoWriter(target_name, fourcc, fps, (width, height), 0)

        vidcap_first = cv2.VideoCapture(filename)
        success, frame1 = vidcap_first.read()
        if success:
            frame1 = frame1
        vidcap_first.release()
        # ret, frame1 = vidcap.read()
        prev_frame = cv2.cvtColor(cv2.resize(frame1, size), cv2.COLOR_BGR2GRAY)

        prev_rgb = None
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 0  # 255

        ii = 0
        # Create two lists for storing optical flow velocity values
        xvel, yvel = [], []

        while vidcap.isOpened():
            ret, frame2 = vidcap.read()
            xsum, ysum = 0, 0

            if ret is not True:
                break

            if ret is True:
                next_frame = cv2.cvtColor(cv2.resize(frame2, size), cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, pyr_scale, levels,
                                                    winsize, iterations, poly_n, poly_sigma, flags)

                if velocity:
                    # Cumulative sum of optical flow vectors
                    for y in range(0, flow.shape[0]):
                        for x in range(0, flow.shape[1]):
                            fx, fy = flow[y, x]
                            xsum += fx
                            ysum += fy

                    # Compute average velocity of pixels by dividing the cumulative sum of optical flow vectors by timesteps
                    xvel.append(
                        self.get_velocity(flow, xsum, flow.shape[1], distance, timestep, move_step, angle_of_view))
                    yvel.append(
                        self.get_velocity(flow, ysum, flow.shape[0], distance, timestep, move_step, angle_of_view))

                else:
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv[..., 0] = ang * 180 / np.pi / 2
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                    img = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                    # techniques on the input image all pixels value above 120 will be set to 255
                    ret, rgb = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

                    if skip_empty:
                        if np.sum(rgb) > 0:
                            out.write(rgb.astype(np.uint8))
                        else:
                            if ii == 0:
                                out.write(rgb.astype(np.uint8))
                            else:
                                out.write(prev_rgb.astype(np.uint8))
                    else:
                        out.write(rgb.astype(np.uint8))

                    if skip_empty:
                        if np.sum(rgb) > 0 or ii == 0:
                            prev_rgb = rgb
                    else:
                        prev_rgb = rgb

                prev_frame = next_frame

            else:
                pb.progress(length)
                break

            pb.progress(ii)
            ii += 1

        vidcap.release()
        out.release()
        return target_name


class MgProgressbar():
    """
    Calls in a loop to create terminal progress bar.
    """

    def __init__(
            self,
            total=100,
            time_limit=0.5,
            prefix='Progress',
            suffix='Complete',
            decimals=1,
            length=40,
            fill='█'):
        """
        Initialize the MgProgressbar object.

        Args:
            total (int, optional): Total iterations. Defaults to 100.
            time_limit (float, optional): The minimum refresh rate of the progressbar in seconds. Defaults to 0.5.
            prefix (str, optional): Prefix string. Defaults to 'Progress'.
            suffix (str, optional): Suffix string. Defaults to 'Complete'.
            decimals (int, optional): Positive number of decimals in process percent. Defaults to 1.
            length (int, optional): Character length of the status bar. Defaults to 40.
            fill (str, optional): Bar fill character. Defaults to '█'.
        """

        self.total = total - 1
        self.time_limit = time_limit
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.now = self.get_now()
        self.finished = False
        self.could_not_get_terminal_window = False
        self.tw_width = 0
        self.tw_height = 0
        self.display_only_percent = False

    def get_now(self):
        """
        Gets the current time.

        Returns:
            datetime.datetime.timestamp: The current time.
        """
        from datetime import datetime
        return datetime.timestamp(datetime.now())

    def over_time_limit(self):
        """
        Checks if we should redraw the progress bar at this moment.

        Returns:
            bool: True if equal or more time has passed than `self.time_limit` since the last redraw.
        """
        callback_time = self.get_now()
        return callback_time - self.now >= self.time_limit

    def adjust_printlength(self):
        if self.tw_width <= 0:
            return
        elif self.could_not_get_terminal_window:
            return
        else:
            _length_before = self.length
            current_length = len(self.prefix) + self.length + \
                self.decimals + len(self.suffix) + 10

            # if the length of printed line is longer than the terminal window's width
            if current_length > self.tw_width:
                diff = current_length - self.tw_width

                # if the difference is shorter than the progress bar length
                if diff < self.length:
                    self.length -= diff  # shorten the progress bar

                # if the difference is at least as long as the progress bar or longer
                else:  # remove suffix
                    current_length = current_length - \
                        len(self.suffix)  # remove suffix
                    diff = current_length - self.tw_width  # recalculate difference

                    # if the terminal width is long enough without suffix
                    if diff <= 0:
                        self.suffix = ""  # just remove suffix

                    # the terminal window is too short even without suffix
                    # remove suffix and test again
                    else:
                        self.suffix = ""

                        # --- SUFFIX IS REMOVED ---

                        # if the difference is shorter than the progress bar
                        if diff < self.length:
                            self.length -= diff  # shorten progress bar

                        # if the difference is longer than the progress bar, remove prefix
                        else:  # remove prefix
                            current_length = current_length - len(self.prefix)
                            diff = current_length - self.tw_width

                            # if the terminal width is long enough without prefix
                            if diff <= 0:
                                self.prefix = ""  # just remove prefix

                            # the terminal window is too short even without prefix (and suffix)
                            # remove prefix and test again
                            else:
                                self.prefix = ""

                                # --- PREFFIX IS REMOVED ---

                                # if the difference is shorter than the progress bar
                                if diff < self.length:
                                    self.length -= diff  # shorten progress bar

                                else:  # display only percent
                                    self.display_only_percent = True

    def progress(self, iteration):
        """
        Progresses the progress bar to the next step.

        Args:
            iteration (float): The current iteration. For example, the 57th out of 100 steps, or 12.3s out of the total 60s.
        """
        if self.finished:
            return
        import sys
        import shutil

        if not self.could_not_get_terminal_window:
            self.tw_width, self.tw_height = shutil.get_terminal_size((0, 0))
            if self.tw_width + self.tw_height == 0:
                self.could_not_get_terminal_window = True
            else:
                self.adjust_printlength()  # this line cannot be tested :'(

        capped_iteration = iteration if iteration <= self.total else self.total
        # Print New Line on Complete
        if iteration >= self.total:
            self.finished = True
            percent = ("{0:." + str(self.decimals) + "f}").format(100)
            filledLength = int(round(self.length))
            bar = self.fill * filledLength
            sys.stdout.flush()
            if self.display_only_percent:
                sys.stdout.write('\r%s' % (percent))
            else:
                sys.stdout.write('\r%s |%s| %s%% %s' %
                                 (self.prefix, bar, percent, self.suffix))
            print()
        elif self.over_time_limit():
            self.now = self.get_now()
            percent = ("{0:." + str(self.decimals) + "f}").format(100 *
                                                                  (capped_iteration / float(self.total)))
            filledLength = int(self.length * capped_iteration // self.total)
            bar = self.fill * filledLength + '-' * (self.length - filledLength)
            sys.stdout.flush()
            if self.display_only_percent:
                sys.stdout.write('\r%s' % (percent))
            else:
                sys.stdout.write('\r%s |%s| %s%% %s' %
                                 (self.prefix, bar, percent, self.suffix))
        else:
            return

    def __repr__(self):
        return "MgProgressbar"


def generate_outfilename(requested_name):
    """Returns a unique filepath to avoid overwriting existing files. Increments requested
    filename if necessary by appending an integer, like "_0" or "_1", etc to the file name.

    Args:
        requested_name (str): Requested file name as path string.

    Returns:
        str: If file at requested_name is not present, then requested_name, else an incremented filename.
    """
    import os
    requested_name = os.path.abspath(requested_name).replace('\\', '/')
    req_of, req_fex = os.path.splitext(requested_name)
    req_of = req_of.replace('\\', '/')
    req_folder = os.path.dirname(requested_name).replace('\\', '/')
    req_of_base = os.path.basename(req_of)
    req_file_base = os.path.basename(requested_name)
    out_increment = 0
    files_in_folder = os.listdir(req_folder)
    # if the target folder is empty, return the requested path
    if len(files_in_folder) == 0:
        return requested_name
    # filter files with same ext
    files_w_same_ext = list(filter(lambda x: os.path.splitext(x)[
                            1] == req_fex, files_in_folder))
    # if there are no files with the same ext
    if len(files_w_same_ext) == 0:
        return requested_name
    # filter for files with same start and ext
    files_w_same_start_ext = list(
        filter(lambda x: x.startswith(req_of_base), files_w_same_ext))
    # if there are no files with the same start and ext
    if len(files_w_same_start_ext) == 0:
        return requested_name
    # check if requested file is already present
    present = None
    try:
        ind = files_w_same_start_ext.index(req_file_base)
        present = True
    except ValueError:
        present = False
    # if requested file is not present
    if not present:
        return requested_name
    # if the original filename is already taken, check if there are incremented filenames
    files_w_increment = list(filter(lambda x: x.startswith(
        req_of_base+"_"), files_w_same_start_ext))
    # if there are no files with increments
    if len(files_w_increment) == 0:
        return f'{req_of}_0{req_fex}'
    # parse increments, discard the ones that are invalid, increment highest
    for file in files_w_increment:
        _of = os.path.splitext(file)[0]
        _only_incr = _of[len(req_of_base)+1:]
        try:
            found_incr = int(_only_incr)
            found_incr = max(0, found_incr)  # clip at 0
            out_increment = max(out_increment, found_incr+1)
        except ValueError:  # if cannot be converted to int
            pass
    # return incremented filename
    return f'{req_of}_{out_increment}{req_fex}'
