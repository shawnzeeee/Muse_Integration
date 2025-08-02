import time
from pylsl import StreamInlet, resolve_streams
import subprocess
import numpy as np
from muselsl.stream import stream
from muselsl.stream import list_muses
import threading

stream_process = None

import asyncio
def get_devices_list():
    print("[INFO] Getting available Muse Devices.")
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        muses = list_muses()
        return muses
    finally:
        asyncio.set_event_loop(None)
    
    
def start_muse_stream(MAC_ADDRESS, start_event, stop_event):
    print("[INFO] Starting muselsl stream...")
    global stream_process
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        stream(MAC_ADDRESS, start_event, stop_event)
    finally:
        asyncio.set_event_loop(None)
    

def muse_connected():
    streams = resolve_streams()
    for stream in streams:
        if stream.type() == 'EEG' and 'Muse' in stream.name():
            print(f"[CONNECTED] pylsl found stream: {stream.name()}")
            return True
    return False

def wait_for_stream(timeout=10):
    print("[INFO] pylsl waiting for EEG stream...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if muse_connected():
            return True
        time.sleep(1)
    raise TimeoutError("[ERROR] Timed out waiting for Muse EEG stream.")

def connect_to_eeg_stream():
    streams = resolve_streams()
    if not streams:
        raise RuntimeError("[ERROR] pylsl could not find EEG stream. Is Muse powered on?")
    print("[INFO] pylsl found muselsl EEG stream.")
    return StreamInlet(streams[0])

eeg_buffer = None
eeg_buffer_lock = threading.Lock()

def update_eeg_buffer(start_event, stop_event):
    global eeg_buffer
    buffer_size = 4 * 250 *2
    eeg_buffer = np.zeros(buffer_size, dtype=np.float32)
    try:
        wait_for_stream(timeout=10)
        inlet = connect_to_eeg_stream()
        start_event.set()
        while not stop_event.is_set():
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            with eeg_buffer_lock:
                eeg_buffer = np.roll(eeg_buffer, -4)      # Shift left by 4
                eeg_buffer[-4:] = sample[:4]              # Insert new sample at the end (right)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    except:
        print("\n Something went wrong creating EEG buffer")
    return "Stream Ended"


from blink_collection import get_classification

def blink_timestamping(start_event, stop_event):
    import csv
    global eeg_buffer
    buffer_size = 4 * 250 * 2
    eeg_buffer = np.zeros(buffer_size, dtype=np.float32)
    csv_path = "blink_data.csv"
    previous_class = 0
    try:
        wait_for_stream(timeout=10)
        inlet = connect_to_eeg_stream()
        start_event.set()
        with open(csv_path, mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ch1", "ch2", "ch3", "ch4", "classification"])
            while not stop_event.is_set():
                sample, timestamp = inlet.pull_sample(timeout=1.0)
                if sample is not None:
                    classification = get_classification()
                    send_code = 0
                    if classification != previous_class:
                        send_code = classification
                    previous_class = classification
                    row = [sample[0], sample[1], sample[2], sample[3], send_code]
                    writer.writerow(row)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    except Exception as e:
        print(f"\n Something went wrong creating EEG buffer: {e}")
    return "Stream Ended"


def get_eeg_buffer():
    global eeg_buffer
    return eeg_buffer

def main():
    muses = list_muses()
    print(muses)


if __name__ == "__main__":
    main()



