import librosa
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy import signal
from pydub import AudioSegment
import librosa.display

# A global flag to track if plots are visible
plots_Visible = False


# GUI window
root = tk.Tk()
root.title("Acoustic Modeling Module.")

# Add a figure to hold RT60 plots
# Add a global variable to track the current plot type
current_plot_type = 'all'  # Initialize to 'all' to show all plots initially
fig, axes = plt.subplots(1, 1, figsize=(10, 5))
axes.set_title("Low, Mid, and High Frequency RT60 Values.")
axes.set_xlabel("Samples")
axes.set_ylabel("RT60 (Seconds)")

# Initialize global variables at the start
rt60_low = float('nan')  # Use NaN to indicate uninitialized values
rt60_mid = float('nan')
rt60_high = float('nan')

# Convert any audio file to .wav
def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.rsplit('.', 1)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

# Display the waveform
def display_waveform(audio, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()


# Function to process audio file and calculate RT60 for low, mid, and high frequency ranges
def process_Audio(file_path):
    global plots_Visible

    # Load the audio file and convert to mono if necessary
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    if len(audio.shape) > 1:  # Check for multi-channel audio
        audio = np.mean(audio, axis=1) # Average the channels

    # Calculate and display the duration of the audio in seconds
    duration = len(audio) / sr  # Duration in seconds
    duration_label.config(text=f"Duration: {duration:.2f} seconds")

    # Display the waveform of the audio file
    display_waveform(audio, sr)

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
    b, a = signal.butter(4, [low, high], btype='band')
    
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
    global axes, current_plot_type

    axes.clear()

    if current_plot_type == 'all':
        # Update the RT60 bar plots for all frequency ranges
        frequency_ranges = ['Low (20-200Hz)', 'Mid (200-2000Hz)', 'High (2000-20000Hz)']
        RT60_Values = [rt60_low, rt60_mid, rt60_high]
        axes.bar(frequency_ranges, RT60_Values, color=['blue', 'green', 'red'])
    elif current_plot_type == 'low':
        # Update the RT60 plot for low frequency range
        frequency_ranges = ['Low (20-200Hz)']
        RT60_Values = [rt60_low]
        axes.bar(frequency_ranges, RT60_Values, color=['blue'])
    elif current_plot_type == 'mid':
        # Update the RT60 plot for mid frequency range
        frequency_ranges = ['Mid (200-2000Hz)']
        RT60_Values = [rt60_mid]
        axes.bar(frequency_ranges, RT60_Values, color=['green'])
    elif current_plot_type == 'high':
        # Update the RT60 plot for high frequency range
        frequency_ranges = ['High (2000-20000Hz)']
        RT60_Values = [rt60_high]
        axes.bar(frequency_ranges, RT60_Values, color=['red'])

    axes.set_title("RT60 for Low, Mid, and High Frequencies.")
    axes.set_ylabel("RT60 (Seconds)")

    plt.draw()  # Redraw the figure

# Function to toggle the visibility of RT60 plots
def toggle_plots(current_plot):
    global current_plot_type, rt60_low, rt60_mid, rt60_high

    if current_plot_type == 'all':
        current_plot_type = 'low'
    elif current_plot_type == 'low':
        current_plot_type = 'mid'
    elif current_plot_type == 'mid':
        current_plot_type = 'high'
    elif current_plot_type == 'high':
        current_plot_type = 'all'

    update_Plots(rt60_low, rt60_mid, rt60_high)  # Update the plots with the new plot type

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
    global rt60_low, rt60_mid, rt60_high
    file_Path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3;*.aac")])
    if file_Path:
        # Update the label with the loaded file name
        file_label.config(text=f"Loaded file: {file_Path.split('/')[-1]}")
        
        # Process the file (convert to WAV if necessary)
        if not file_Path.endswith(".wav"):
            file_Path = convert_to_wav(file_Path)  # Assuming you implemented the conversion logic
        rt60_Low, rt60_Mid, rt60_High = process_Audio(file_Path)
        report = RT60_Report(rt60_Low, rt60_Mid, rt60_High)
        print(report)

# Create the Load button to load an audio file
load_button = tk.Button(root, text="Load Audio File", command=load_File)
load_button.pack()

# Create the Toggle Plot button to toggle visibility of RT60 plots
toggle_button = tk.Button(root, text="Toggle RT60 Plots", command=toggle_plots)
toggle_button.pack()

# Add a label to display the loaded file name
file_label = tk.Label(root, text="No file loaded", font=("Arial", 12))
file_label.pack()

# Add duration of audio label
duration_label = tk.Label(root, text="Duration: 0.00 seconds", font=("Arial", 12))
duration_label.pack()

#
combined_button = tk.Button(root, text="Show Combined Plots", command=lambda: toggle_plots('all'))
combined_button.pack()

# Start the GUI main loop
root.mainloop()
