
import json
import numpy as np
import time
from pathlib import Path

from spikeinterface.extractors import read_mearec
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion_estimation import estimate_motion

from spikeinterface.widgets import plot_probe_map

import scipy.interpolate

import matplotlib.pyplot as plt

import MEArec as mr

class BenchmarkMotionEstimationMearec:
    
    def __init__(self, mearec_filename, 
                title='',
                detect_kwargs={},
                select_kwargs=None,
                localize_kwargs={},
                estimate_motion_kwargs={},
                output_folder=None,
                job_kwargs={'chunk_duration' : '1s', 'n_jobs' : -1, 'progress_bar':True, 'verbose' :True}, 
                overwrite=False):
                
        self.mearec_filename = mearec_filename
        self.recording, self.gt_sorting = read_mearec(self.mearec_filename)
        self.title = title
        
        self.job_kwargs = job_kwargs
        self.detect_kwargs = detect_kwargs
        self.select_kwargs = select_kwargs
        self.localize_kwargs = localize_kwargs
        self.estimate_motion_kwargs = estimate_motion_kwargs
        self.overwrite = overwrite

        self.output_folder = output_folder


    def run(self):

        if self.output_folder is not None:
            if self.output_folder.exists() and not self.overwrite:
                print(f"The folder {self.output_folder} is not empty")
                return

        t0 = time.perf_counter()
        self.peaks = detect_peaks(self.recording, **self.detect_kwargs, **self.job_kwargs)
        t1 = time.perf_counter()
        if self.select_kwargs is not None:
            self.selected_peaks = select_peaks(self.peaks, **self.select_kwargs, **self.job_kwargs)
        else:
            self.selected_peaks = self.peaks
        t2 = time.perf_counter()
        self.peak_locations = localize_peaks(self.recording, self.selected_peaks, **self.localize_kwargs, **self.job_kwargs)
        t3 = time.perf_counter()
        self.motion, self.temporal_bins, self.spatial_bins = estimate_motion(self.recording, self.selected_peaks, self.peak_locations, 
                                        **self.estimate_motion_kwargs)

        ## You were right, we need to subtract the first value of the motion to have something
        ## properly centered. Otherwise, there is a bias distorting traces
        # self.motion -= self.motion[0]
        t4 = time.perf_counter()

        self.run_times = dict(
            detect_peaks=t1 -t0,
            select_peaks=t2 - t1,
            localize_peaks=t3 - t2,
            estimate_motion=t4 - t3,
        )

        self.compute_gt_motion()

        ## save folder
        if self.output_folder is not None:
            self.save_to_folder(self.output_folder)


    def compute_gt_motion(self):
        self.gt_unit_positions, _ = mr.extract_units_drift_vector(self.mearec_filename, time_vector=self.temporal_bins)
        unit_motions = self.gt_unit_positions - self.gt_unit_positions[0, :]
        unit_positions = np.mean(self.gt_unit_positions, axis=0)

        if self.spatial_bins is None:
            self.gt_motion = np.mean(unit_motions, axis=1)[:, None]
            channel_positions = self.recording.get_channel_locations()
            probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()
            center = (probe_y_min + probe_y_max)//2
            self.spatial_bins = np.array([center])
        else:
            # time, units
            self.gt_motion = np.zeros_like(self.motion)
            for t in range(self.gt_unit_positions.shape[0]):
                f = scipy.interpolate.interp1d(unit_positions, unit_motions[t, :], fill_value="extrapolate")
                self.gt_motion[t, :] = f(self.spatial_bins)


    _array_names = ('gt_unit_positions', 'peaks', 'selected_peaks', 'motion', 'temporal_bins', 
                    'spatial_bins', 'peak_locations', 'gt_motion')
    
    _dict_kwargs_names = ('job_kwargs', 'detect_kwargs', 'select_kwargs', 'localize_kwargs', 'estimate_motion_kwargs')

    def save_to_folder(self, folder):

        if folder.exists():
            if self.overwrite:
                import shutil
                shutil.rmtree(folder)
        folder.mkdir(parents=True)

        folder = Path(folder)

        info = {
            'mearec_filename': str(self.mearec_filename),
            'title': self.title,
        }
        (folder / 'info.json').write_text(json.dumps(info, indent=4), encoding='utf8')

        for name in self._array_names:
            value = getattr(self, name)
            if value is not None:
                np.save(folder / f'{name}.npy', value)
        
        for name in self._dict_kwargs_names:
            d = getattr(self, name)
            file = folder / f'{name}.json'
            if d is not None:
                file.write_text(json.dumps(d, indent=4), encoding='utf8')

        run_times_filename = folder / 'run_times.json'
        run_times_filename.write_text(json.dumps(self.run_times, indent=4),encoding='utf8')

    @classmethod
    def load_from_folder(cls, folder):
        folder = Path(folder)
        assert folder.exists()

        with open(folder / 'info.json', 'r') as f:
            info = json.load(f)
        mearec_filename = info['mearec_filename']
        title = info['title']

        dict_kwargs = dict()
        for name in cls._dict_kwargs_names:
            filename = folder / f'{name}.json' 
            if filename.exists():
                with open(filename, 'r') as f:
                    d = json.load(f)
            else:
                d = None
            dict_kwargs[name] = d

        bench = cls(mearec_filename, output_folder=folder, title=title, overwrite=False, **dict_kwargs)
        for name in cls._array_names:
            filename = folder / f'{name}.npy'
            if filename.exists():
                arr = np.load(filename)
            else:
                arr = None
            setattr(bench, name, arr)

        with open(folder / 'run_times.json', 'r') as f:
            bench.run_times = json.load(f)

        return bench


    def plot_drift(self, scaling_probe=1.5):
                
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(1, 3)

        ax = fig.add_subplot(gs[0])
        plot_probe_map(self.recording, ax=ax)
        _simpleaxis(ax)

        mr_recording = mr.load_recordings(self.mearec_filename)
            
        for loc in mr_recording.template_locations:
            if len(mr_recording.template_locations.shape) == 3:
                ax.plot([loc[0, 1], loc[-1, 1]], [loc[0, 2], loc[-1, 2]], alpha=0.7, lw=2)
            else:
                ax.scatter([loc[1]], [loc[2]], alpha=0.7, s=100)
    
        ymin, ymax = ax.get_ylim()
        ax.set_ylabel('depth (um)')
        ax.set_xlabel('depth (um)')

        channel_positions = self.recording.get_channel_locations()
        probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()

        ax.set_ylim(scaling_probe*probe_y_min, scaling_probe*probe_y_max)

        ax = fig.add_subplot(gs[1:3])
        for i in range(self.gt_unit_positions.shape[1]):
            ax.plot(self.temporal_bins, self.gt_unit_positions[:, i], alpha=0.5, ls='--')
        
        for i in range(self.gt_motion.shape[1]):
            depth = self.spatial_bins[i]
            ax.plot(self.temporal_bins, self.gt_motion[:, i] + depth, color='red', lw=2)


        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('time (s)')
        _simpleaxis(ax)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.set_ylim(scaling_probe*probe_y_min, scaling_probe*probe_y_max)
        xmin, xmax = ax.get_xlim()
        ax.plot([xmin, xmax], [probe_y_min, probe_y_min], 'k--', alpha=0.5)
        ax.plot([xmin, xmax], [probe_y_max, probe_y_max], 'k--', alpha=0.5)


    def plot_peaks(self, scaling_probe=1.5, show_drift=True):

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(1, 3)
        # Create the Axes.

        ax = fig.add_subplot(gs[0])
        plot_probe_map(self.recording, ax=ax)
        _simpleaxis(ax)

        ymin, ymax = ax.get_ylim()
        ax.set_ylabel('depth (um)')
        ax.set_xlabel('depth (um)')

        channel_positions = self.recording.get_channel_locations()
        probe_y_min, probe_y_max = channel_positions[:, 1].min(), channel_positions[:, 1].max()

        ax.set_ylim(scaling_probe*probe_y_min, scaling_probe*probe_y_max)

        ax = fig.add_subplot(gs[1:3])
        x = self.selected_peaks['sample_ind']/self.recording.get_sampling_frequency()
        y = self.peak_locations['y']
        ax.scatter(x, y, s=1, color='k', alpha=0.05)
        
        xmin, xmax = ax.get_xlim()
        ax.plot([xmin, xmax], [probe_y_min, probe_y_min], 'k--', alpha=0.5)
        ax.plot([xmin, xmax], [probe_y_max, probe_y_max], 'k--', alpha=0.5)

        _simpleaxis(ax)
        ax.set_yticks([])
        ax.set_ylim(scaling_probe*probe_y_min, scaling_probe*probe_y_max)
        ax.spines['left'].set_visible(False)
        ax.set_xlabel('time (s)')

        if show_drift:
            if self.spatial_bins is None:
                center = (probe_y_min + probe_y_max)//2
                ax.plot(self.temporal_bins, self.gt_motion[:, 0] + center, color='red', lw=2)
            else:
                for i in range(self.gt_motion.shape[1]):
                    depth = self.spatial_bins[i]
                    ax.plot(self.temporal_bins, self.gt_motion[:, i] + depth, color='red', lw=2)


    def estimation_vs_depth(self):
        fig, axes = plt.subplots(2, figsize=(15,10))

        corrs = {}

        duration = self.recording.get_total_duration()
        
        ax = axes[0]
        ax.plot(self.temporal_bins, self.gt_motion, lw=2, c='b')
        ax.plot(self.temporal_bins, self.motion.mean(1), lw=2, c='r')
        ax.fill_between(self.temporal_bins, self.motion.mean(1)-self.motion.std(1), 
                                self.motion.mean(1) + self.motion.std(1), color='r', alpha=0.25)
        
        ax.set_ylabel('drift (um)')
        ax.set_xlabel('time (s)')
        _simpleaxis(ax)
        
        corrs = []
        for i in range(self.motion.shape[1]):
            corrs += [np.corrcoef(self.motion[:, i], self.gt_motion[:, i])[0,1]]

        ax = axes[1]
        ax.scatter(self.spatial_bins, corrs)
        ax.set_ylabel('Correlation between drift')
        ax.set_xlabel('depth (um)')
        _simpleaxis(ax)

    def view_errors(self):
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)

        ax = fig.add_subplot(gs[0, :])
        im = ax.imshow(np.abs(self.gt_motion - self.motion).T, aspect='auto', interpolation='nearest', origin='lower', 
        extent=(self.temporal_bins[0], self.temporal_bins[-1], self.spatial_bins[0], self.spatial_bins[-1]))
        plt.colorbar(im, ax=ax, label='error')
        ax.set_ylabel('depth (um)')
        ax.set_xlabel('time (s)')

        ax = fig.add_subplot(gs[1, 0])
        mean_error = np.linalg.norm(self.gt_motion - self.motion, axis=1)
        ax.plot(self.temporal_bins, mean_error)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('error')
        _simpleaxis(ax)

        ax = fig.add_subplot(gs[1, 1])
        depth_error = np.linalg.norm(self.gt_motion - self.motion, axis=0)
        ax.plot(self.spatial_bins, depth_error)
        ax.set_xlabel('depth (um)')
        ax.set_ylabel('error')
        _simpleaxis(ax)

def _simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()