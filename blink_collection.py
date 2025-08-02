# --- Blink Prompt UI ---
import tkinter as tk
import time
import threading


gesture_code_lock = threading.Lock()
gesture_code = 0


def send_blink_classification(code):
    global gesture_code, gesture_code_lock
    with gesture_code_lock:
        gesture_code = code
def get_classification():
    global gesture_code, gesture_code_lock
    with gesture_code_lock:
        return gesture_code

def blink_prompt(stop_event):
    def run_blink_sequence():
        num_blinks=10
        for i in range(num_blinks):
            # Countdown before blink (also serves as relax period)
            send_blink_classification(0)  # 0 = no blink
            for sec in range(3, 0, -1):
                label.config(text=f"Blink in {sec}... ({i+1}/{num_blinks})")
                root.update()
                time.sleep(1)
            label.config(text=f"Blink NOW! ({i+1}/{num_blinks})")
            root.update()
            send_blink_classification(1)  # 1 = blink
            time.sleep(0.1)
        stop_event.set()
        label.config(text="Done! Thank you.")
        root.update()

    root = tk.Tk()
    root.title("Blink Prompt")
    root.geometry("300x100")
    label = tk.Label(root, text="Get ready to blink!", font=("Arial", 16))
    label.pack(expand=True)
    threading.Thread(target=run_blink_sequence, daemon=True).start()
    root.mainloop()

if __name__ == "__main__":
    blink_prompt()