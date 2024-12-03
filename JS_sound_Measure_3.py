import librosa
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from spicy import signal

# A global flag to track if plots are visible
plots_Visible = False

# GUI window
root = tk.Tk()
root.title("Acoustic Modeling Module.")

# Add a figure to hold RT60 plots
fig, axes = plt.subplots(1, 1, figsize=(10, 5))
axes.set_title("Low, Mid, and High Frequency RT60 Values.")
axes.set_xlabel("Samples")
axes.set_ylabel("RT60 (Seconds)")

# Function to process audio file and calculate RT60 for low, mid, and high frequency ranges
def process_Audio(file_path):
    global plots_Visible

    # Load the audio file and convert to mono if necessary
    audio = (librosa.load(file_path, sr=None, mono=True))
    sr = (librosa.load(file_path, sr=None, mono=True))

    # Calculate RT60 for low, mid, and high frequency ranges
    rt60_low = compute_rt60(audio, sr, low_freq=20, high_freq=200)
    rt60_mid = compute_rt60(audio, sr, low_freq=200, high_freq=2000)
    rt60_high = compute_rt60(audio, sr, low_freq=2000, high_freq=20000)

    # If plots are visible, update them
    if plots_Visible:
        update_Plots(rt60_low, rt60_mid, rt60_high)
    return rt60_low, rt60_mid, rt60_high

def compute_rt60(audio_signal, sr, low_freq, high_freq):
    # Apply bandpass filter
    nyquist = sr / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b = signal.butter(4, [low, high], btype='band')
    a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, audio_signal)

    # Compute the energy decay curve (EDC)
    envelope = np.abs(filtered_signal) ** 2
    decay = -10 * np.log10(envelope / np.max(envelope))

    # Finds time when the signal decays by 60 dB (RT60).
    time = np.arange(len(decay)) / sr
    rt60_index = np.where(decay >= 60)[0]
    rt60_time = time[rt60_index[0]] if len(rt60_index) > 0 else time[-1]

    return rt60_time

# Function to update RT60 plots
def update_Plots(rt60_low, rt60_mid, rt60_high):
    global axes

    #Clears previous plots
    axes.clear()

    # Update the RT60 bar plots
    frequency_ranges = ['Low (20-200Hz)', 'Mid (200-2000Hz)', 'High (2000-20000Hz)']
    RT60_Values = [rt60_low, rt60_mid, rt60_high]
    axes.bar(frequency_ranges, RT60_Values, color= ['blue', 'green', 'red'])

    axes.set_title("RT60 for Low, Mid, and High Frequencies.")
    axes.set_ylabel("RT60 (Seconds)")

    plt.draw()  # Redraw the figure

# Function to toggle the visibility of RT60 plots
def toggle_plots():
    global plots_Visible, fig
    plots_Visible = not plots_Visible

    # If plots are visible, show them; otherwise, hide them
    if plots_Visible:
        plt.show()
    else:
        plt.close(fig)

# Function to generate a simple report on RT60 values
def RT60_Report(rt60_Low, rt60_Mid, rt60_High):
    target_rt60 = 0.5  # Target RT60 for intelligibility
    report_Output = f"RT60 Measurements:\n"
    report_Output += f"Low Frequencies (20-200 Hz): {rt60_Low:.3f} seconds\n"
    report_Output += f"Mid Frequencies (200-2000 Hz): {rt60_Mid:.3f} seconds\n"
    report_Output += f"High Frequencies (2000-20000 Hz): {rt60_High:.3f} seconds\n"

    diff_Low = (target_rt60 - rt60_Low)
    diff_Mid = (target_rt60 - rt60_Mid)
    diff_High = (target_rt60 - rt60_High)

    report_Output += f"\nRT60 Differences from Target (≤ 0.5 seconds):\n"
    report_Output += f"Low Frequency Difference: {diff_Low:.3f} seconds\n"
    report_Output += f"Mid Frequency Difference: {diff_Mid:.3f} seconds\n"
    report_Output += f"High Frequency Difference: {diff_High:.3f} seconds\n"

    return report_Output

# Function to handle loading the audio file
def load_File():
    file_Path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3;*.aac")])
    if file_Path:
        RT60_Low, RT60_Mid, Rt60_High = process_Audio(file_Path)
        report = RT60_Report(RT60_Low, RT60_Mid, Rt60_High)
        print(report)

# Create the Load button to load an audio file
load_button = tk.Button(root, text="Load Audio File", command=load_File)
load_button.pack()

# Create the Toggle Plot button to toggle visibility of RT60 plots
toggle_button = tk.Button(root, text="Toggle RT60 Plots", command=toggle_plots)
toggle_button.pack()

# Start the GUI main loop
root.mainloop()