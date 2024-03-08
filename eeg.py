import sys
import time
import logging
from time import sleep
from multiprocessing import Process

import numpy as np
import pandas as pd

from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams
from muselsl import stream, list_muses, record, constants as mlsl_cnsts
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_byprop

from eegnb.devices.utils import (
    get_openbci_usb,
    create_stim_array,
    SAMPLE_FREQS,
    EEG_INDICES,
    EEG_CHANNELS,
)


logger = logging.getLogger(__name__)


class EEG:
    device_name: str
    stream_started: bool = False

    def __init__(self,device='muse2',):
        # determine if board uses brainflow or muselsl backend
        self.device_name = device
        self.stream_started = False
        self._init_muselsl()
        self.get_recent()  # run this at initialization to get some


    def _init_muselsl(self):
        """Initialize the MuseLSL backend."""
        # Assuming LSL stream is already being broadcasted by another app (e.g., Muse Direct or BlueMuse)
        streams = resolve_byprop('type', 'EEG', timeout=2)
        if streams:
            self.inlet = StreamInlet(streams[0], max_chunklen=12)
        else:
            logger.error("Can't find EEG stream.")



    def start(self, duration):
        if sys.platform in ["linux", "linux2", "darwin"]:
            # Look for muses
            self.muses = list_muses()
            # self.muse = muses[0]

            # Start streaming process
            self.stream_process = Process(
                target=stream, args=(self.muses[0]["address"],)
            )
            self.stream_process.start()

        # Create markers stream outlet
        self.muse_StreamInfo = StreamInfo(
            "Markers", "Markers", 1, 0, "int32", "myuidw43536"
        )
        self.muse_StreamOutlet = StreamOutlet(self.muse_StreamInfo)

        # Start a background process that will stream data from the first available Muse
        print("starting background recording process")
        if self.save_fn:
            print("will save to file: %s" % self.save_fn)
        self.recording = Process(target=record, args=(duration, self.save_fn))
        self.recording.start()

        time.sleep(5)
        self.stream_started = True
        self.push_sample([99], timestamp=time.time())

    def stop(self):
        pass

    def push_sample(self, marker, timestamp):
        self.muse_StreamOutlet.push_sample(marker, timestamp)

    def get_recent(self, n_samples: int = 256, restart_inlet: bool = False):
        if self._muse_recent_inlet and not restart_inlet:
            inlet = self._muse_recent_inlet
        else:
            # Initiate a new lsl stream
            streams = resolve_byprop("type", "EEG", timeout=mlsl_cnsts.LSL_SCAN_TIMEOUT)
            if not streams:
                raise Exception("Couldn't find any stream, is your device connected?")
            inlet = StreamInlet(streams[0], max_chunklen=mlsl_cnsts.LSL_EEG_CHUNK)
            self._muse_recent_inlet = inlet

        info = inlet.info()
        sfreq = info.nominal_srate()
        description = info.desc()
        n_chans = info.channel_count()

        self.sfreq = sfreq
        self.info = info
        self.n_chans = n_chans

        timeout = (n_samples / sfreq) + 0.5
        samples, timestamps = inlet.pull_chunk(timeout=timeout, max_samples=n_samples)

        samples = np.array(samples)
        timestamps = np.array(timestamps)

        ch = description.child("channels").first_child()
        ch_names = [ch.child_value("label")]
        for i in range(n_chans):
            ch = ch.next_sibling()
            lab = ch.child_value("label")
            if lab != "":
                ch_names.append(lab)

        df = pd.DataFrame(samples, index=timestamps, columns=ch_names)
        return df