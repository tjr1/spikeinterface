import numpy as np
# import pandas as pd
from pathlib import Path
import os

from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.core.core_tools import define_function_from_class

class Mef3RecordingExtractor(BaseRecording):
    """Load Mef3 data as an extractor object.

    MEF3 is the Multiscale Electrophysiology Format version 3.

    pymef and mef_tools are required to use this extractor.

    Parameters
    ----------
    folder_path: str or Path
        Path to .mefd folder.
        Path to electrode locations file (.tsv) (optional)

    Returns
    -------
    recording : Mef3RecordingExtractor
        The recording extractor for the mef3 data
    """

    extractor_name = "Mef3Recording"
    mode = "folder"
    installation_mesg = "To use the Mef3RecordingExtractor, install pymef \n\n pip install pymef\n\n note: pymef requires python 3.8 currently\n\n"
    name = "Mef3"

    @classmethod
    def get_unique_stream_rate_groups(cls, reader) -> list:
        # chans = reader.channels
        chans = reader.read_ts_channel_basic_info()    
        channel_fs = [chans[ch]['fsamp'][0] for ch in range(len(chans))]
        channel_fs = set(channel_fs)
        return channel_fs


    # def get_channel_groups(cls, reader, electrode_locations_file) -> pd.DataFrame:
    #     # [channel_ind:int, channe_id:str, channel_fs:int, 
    #     #  stream_rate_group:str, stream_rate_group_ind:int, 
    #     #  electrode_group:str, electrode_group_ind:int, 
    #     #  electrode_subgroup:str, electrode_subgroup_ind:int]

    #     df = pd.DataFrame(columns = ['channel_ind', 'channel_id', 'channel_fs', 
    #                                  'stream_rate_group', 'stream_rate_group_ind', 
    #                                  'electrode_group', 'electrode_group_ind', 
    #                                  'electrode_subgroup', 'electrode_subgroup_ind'])
        
    #     chans = reader.channels
    #     channel_ids = [chans[i].name for i in range(len(chans))]
    #     channel_fs = [reader.get_property('fsamp', ch) for ch in chans]

        

    def __init__(
        self, 
        folder_path: str, 
        stream_rate: int,
        password: str = None):

        try:
            from pymef.mef_session import MefSession
            # from mef_tools.io import MefReader, MefWriter
        except ImportError:
            raise ImportError(self.installation_mesg)

        # reader
        reader = MefSession(folder_path,password)
        self._reader = reader

        unique_stream_rate_groups = self.get_unique_stream_rate_groups(reader = reader)
        assert stream_rate in unique_stream_rate_groups, (
            f"The `stream_rate` '{stream_rate}' was not found in the available list of stream rates! "
            f"Please choose one of {unique_stream_rate_groups}."
        )

        # get basic metadata
        sampling_frequency = stream_rate # the stream rate you asked for
        chans = reader.read_ts_channel_basic_info()
        self._chans = chans

        start_time = min([chans[ch]['start_time'][0] for ch in range(len(chans))])
        end_time = max([chans[ch]['end_time'][0] for ch in range(len(chans))])
        self._start_time = start_time
        self._end_time = end_time
        chans_in_stream_rate_group = [chans[i] for i in range(len(chans)) if chans[i]['fsamp'][0] == stream_rate]
        self._chans_in_stream_rate_group = chans_in_stream_rate_group

        channel_ids = [chans_in_stream_rate_group[i]['name'] for i in range(len(chans_in_stream_rate_group))]
        channel_gains = [np.round(1/chans_in_stream_rate_group[i]['ufact'][0]) for i in range(len(chans_in_stream_rate_group))]
        channel_offsets = np.zeros(len(channel_ids))
        dtype = float

        times_kwargs = dict(sampling_frequency=sampling_frequency, t_start=start_time)
        self._times_kwargs = times_kwargs

        # init
        BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype)
        self.set_channel_gains(channel_gains)
        self.set_channel_offsets(channel_offsets)
        self.extra_requirements.append("pymef")

        # should add probe info here and set_property

        numframes = chans_in_stream_rate_group[0]['nsamp'][0]
        recording_segment = Mef3RecordingSegment(reader=self._reader, chans_in_stream_rate_group=chans_in_stream_rate_group, times_kwargs=times_kwargs)

        self.add_recording_segment(recording_segment)

        self._kwargs = {
            "folder_path": folder_path,
            "stream_rate": stream_rate,
            "password": password,
        }

class Mef3RecordingSegment(BaseRecordingSegment):
    def __init__(self, reader, chans_in_stream_rate_group, times_kwargs):

        BaseRecordingSegment.__init__(self, **times_kwargs)
        
        self._reader = reader
        self._chans_in_stream_rate_group = chans_in_stream_rate_group
        self._channel_str_list = [self._chans_in_stream_rate_group[ch]['name'] for ch in range(len(self._chans_in_stream_rate_group))]

    def get_num_samples(self):
        # ns = self.reader.get_property('nsamp',self._chans_in_stream_rate_group[0])
        ns = self._chans_in_stream_rate_group[0]['nsamp'][0]
        return ns
    
    def get_traces(self, start_frame: int, end_frame: int, channel_indices: slice) -> np.ndarray:
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()
        traces = np.transpose(np.array(self._reader.read_ts_channels_sample(self._channel_str_list[channel_indices], [[start_frame, end_frame]])))
        return traces
    
read_mef3_recording = define_function_from_class(source_class=Mef3RecordingExtractor, name="read_mef3_recording")

mefE = Mef3RecordingExtractor('/Volumes/HD2/mef3/small_mef3/even_smaller_mef3_v2.mefd',5000,'password2','/Volumes/HD2/mef3/small_mef3/even_smaller_mef3_v2.mefd/sub-MSEL02545_ses-ieeg01_electrodes.tsv')
print(mefE)
print(mefE.get_unique_stream_rate_groups(mefE._reader))
print(mefE.get_channel_ids())
print(mefE.get_num_frames())
print(mefE.get_num_samples())
print(mefE.get_sampling_frequency())
mef3Data = mefE.get_traces(start_frame=0,end_frame=1000)
print(mef3Data)
print(mef3Data.shape)
print(mefE.get_channel_gains())
print(mefE.get_channel_offsets())

mefSeg = Mef3RecordingSegment(mefE._reader, mefE._chans_in_stream_rate_group, mefE._times_kwargs)
print(mefSeg)
print(mefSeg.get_num_samples())
print(mefSeg.get_times())
mef3Seg = mefSeg.get_traces(start_frame=0,end_frame=1000,channel_indices=slice(0,10))
print(mef3Seg.shape)
print(mefSeg.get_times_kwargs())
print(mefSeg.time_vector)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(mefSeg.get_times(),mef3Seg)
# plt.show()

# todo: timestamps similar to openepys.py
# todo: process probe info
