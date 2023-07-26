import numpy as np
import pandas as pd
from pathlib import Path


from spikeinterface.core import BaseRecording, BaseRecordingSegment
from spikeinterface.core.core_tools import define_function_from_class


class Mef3Extractor(BaseRecording):
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
    recording : Mef3Extractor
        The recording extractor for the mef3 data
    """

    extractor_name = "Mef3"
    installed = HAVE_PYMEF
    mode = "folder"
    installation_mesg = "To use the Mef3Extractor, install pymef \n\n pip install pymef\n\n note: pymef requires python 3.8 currently\n\n"
    name = "Mef3"

    @classmethod
    def get_unique_stream_rate_groups(cls, Rdr) -> list[int]:
        # chans = Rdr.channels
        chans = Rdr.read_ts_channel_basic_info()    
        channel_fs = [chans[ch]['fsamp'][0] for ch in range(len(chans))]
        channel_fs = set(channel_fs)
        return channel_fs


    # def get_channel_groups(cls, Rdr, electrode_locations_file) -> pd.DataFrame:
    #     # [channel_ind:int, channe_id:str, channel_fs:int, 
    #     #  stream_rate_group:str, stream_rate_group_ind:int, 
    #     #  electrode_group:str, electrode_group_ind:int, 
    #     #  electrode_subgroup:str, electrode_subgroup_ind:int]

    #     df = pd.DataFrame(columns = ['channel_ind', 'channel_id', 'channel_fs', 
    #                                  'stream_rate_group', 'stream_rate_group_ind', 
    #                                  'electrode_group', 'electrode_group_ind', 
    #                                  'electrode_subgroup', 'electrode_subgroup_ind'])
        
    #     chans = Rdr.channels
    #     channel_ids = [chans[i].name for i in range(len(chans))]
    #     channel_fs = [Rdr.get_property('fsamp', ch) for ch in chans]

        

    def __init__(
        self, 
        folder_path: str, 
        stream_rate: int,
        pword2: Optional[str] = None,
        electrode_locations_file: Optional[str] = None):

        try:
            from pymef.mef_session import MefSession
            from mef_tools.io import MefReader, MefWriter
        except ImportError:
            raise ImportError(self.installation_mesg)

        folder_path = Path(folder_path)
        assert folder_path.suffix == ".mefd"

        if not(electrode_locations_file is None):
            electrode_locations_file = Path(electrode_locations_file)

        if not(pword2 is None):
            pword2 = ''

        # reader
        Rdr = MefReader(path_input,password2=pword2)
        self._Rdr = Rdr



        unique_stream_rate_groups = self.get_unique_stream_rate_groups(Rdr = Rdr)
        assert stream_rate in unique_stream_rate_groups, (
            f"The `stream_rate` '{stream_rate}' was not found in the available list of stream rates! "
            f"Please choose one of {unique_stream_rate_groups}."
        )

        # get basic metadata
        sampling_frequency = stream_rate
        # start_time = min(Rdr.get_property('start_time'))
        # end_time = max(Rdr.get_property('end_time'))

        chans = ms.read_ts_channel_basic_info()
        start_time = min([chans[ch]['start_time'][0] for ch in range(len(chans))])
        end_time = max([chans[ch]['end_time'][0] for ch in range(len(chans))])
        # chans = Rdr.channels
        # chans_in_stream_rate_group = [chans[i] for i in range(len(chans)) if Rdr.get_property('fsamp', chans[i]) == stream_rate]
        chans_in_stream_rate_group = [chans[i] for i in range(len(chans)) if chans[i]['fsamp'][0] == stream_rate]

        channel_ids = [chans_in_stream_rate_group[i]['name'] for i in range(len(chans_in_stream_rate_group))]
        channel_gains = [np.round(1/chans_in_stream_rate_group[i]['ufact'][0]) for i in range(len(chans_in_stream_rate_group))]
        channel_offsets = np.zeros(len(channel_inds))
        dtype = numpy.ndarray

        # init
        BaseRecording.__init__(self, channel_ids=channel_ids, sampling_frequency=sampling_frequency, dtype=dtype)
        self.set_channel_gains(channel_gains)
        self.set_channel_offsets(channel_offsets)
        self.extra_requirements.append("pymef")
        # self.extra_requirements.append("mef-tools")

        # should add probe info here and set_property


        recording_segment = Mef3Segment(Rdr=self._Rdr, chans_in_stream_rate_group=chans_in_stream_rate_group) # need chans_in_stream_rate_group?

        self.add_recording_segment(recording_segment)

        # self, 
        # folder_path: str, 
        # stream_rate: int,
        # pword2: Optional[str] = None,
        # electrode_locations_file: Optional[str] = None):
    
        self._kwargs = {
            "folder_path": folder_path,
            "stream_rate": stream_rate,
            "pword2": pword2,
            "electrode_locations_file": electrode_locations_file
        }

class Mef3Segment(BaseRecordingSegment):
    def __init__(self, Rdr, chans_in_stream_rate_group):
        BaseRecordingSegment.__init__(self)
        self._Rdr = Rdr
        self._chans_in_stream_rate_group = chans_in_stream_rate_group


    def get_num_samples(self):
        # ns = self.Rdr.get_property('nsamp',self._chans_in_stream_rate_group[0])
        ns = chans_in_stream_rate_group[0]['nsamp'][0]
        return ns
    
    def get_traces(self, start_frame: int, end_frame: int, channel_indices):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.get_num_samples()
        if channel_indices is None:
            channel_indices = slice(None)

        traces = self._Rdr.read(nsel=slice(start_frame, end_frame), volts=False)

        channel_str_list = [self._chans_in_stream_rate_group[ch]['name'] for ch in range(len(self._chans_in_stream_rate_group))]
        traces = self._Rdr.read_ts_channels_sample(channel_str_list[channel_indices];, [[start_frame, end_frame]])

