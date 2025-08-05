from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import threading
import asyncio


tasks = {}



from muse_stream import start_muse_stream, update_eeg_buffer, blink_timestamping
#from classifier import classify
from classifier_csp import classify
from blink_collection import blink_prompt
import time

muselsl_start_event = threading.Event()
muselsl_stop_event = threading.Event()
pylsl_stop_event = threading.Event()
pylsl_start_event = threading.Event()
classifier_thread = None
pylsl_thread = None
muselsl_thread = None
blink_collection_thread = None
def connect_muse(mac_address: str):
    global muselsl_thread, muselsl_start_event, muselsl_stop_event, pylsl_thread, pylsl_stop_event
    pylsl_start_event.clear()
    pylsl_stop_event.clear()
    muselsl_start_event.clear()
    muselsl_stop_event.clear()
    muselsl_thread = threading.Thread(target=start_muse_stream, args=(mac_address, muselsl_start_event, muselsl_stop_event))
    muselsl_thread.start()

    while not muselsl_start_event.is_set() and muselsl_thread.is_alive():
        time.sleep(0.1)

    if muselsl_start_event.is_set():
        pylsl_thread = threading.Thread(target=update_eeg_buffer, args=(pylsl_start_event, pylsl_stop_event))
        pylsl_thread.start()
        while not pylsl_start_event.is_set() and pylsl_thread.is_alive():
            time.sleep(0.1)
        if pylsl_start_event.is_set():
            classifier_thread = threading.Thread(target=classify, args=(muselsl_stop_event,))
            classifier_thread.start()
            # blink_collection_thread = threading.Thread(target=blink_prompt, args=(muselsl_stop_event,))
            # blink_collection_thread.start()
            # blink_collection_thread.join()
            # disconnect_muse()

    return {"data" : "Stream Stopped"}



def disconnect_muse():
    global muselsl_thread, muselsl_stop_event, pylsl_stop_event
    pylsl_stop_event.set()
    if pylsl_thread is not None:
        pylsl_thread.join()  # Wait for the thread to terminate
    muselsl_stop_event.set()
    if muselsl_thread is not None:
        muselsl_thread.join()
    return {"data": "Pylsl and muselsl terminated"}

def main():
    try:
        response = connect_muse('00:55:DA:B0:1E:78')
        print(response)
    except Exception as e:
        print(e)
if __name__ == "__main__":
    main()