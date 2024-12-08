import librosa
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy import signal
from scipy.io import wavfile
from scipy.signal import welch
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

time_curves = {'low': None, 'mid': None, 'high': None}
decay_curves = {'low': None, 'mid': None, 'high': None}

current_plot = None

# Convert any audio file to .wav
def convert_to_wav(file_path):
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.rsplit('.', 1)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

# Display the waveform
def display_waveform(file_path):
    # Read the audio file using scipy.io.wavfile
    samplerate, data = wavfile.read(file_path)
    length = data.shape[0] / samplerate  # Calculate audio duration
    time = np.linspace(0., length, data.shape[0])  # Create time vector

    # Check if the audio is mono or stereo
    plt.figure(figsize=(10, 6))
    if len(data.shape) == 1:  # Mono
        plt.plot(time, data, label="Mono")
        plt.title("Waveform - Mono")
    else:  # Stereo
        plt.plot(time, data[:, 0], label="Left channel")  # Left channel
        plt.plot(time, data[:, 1], label="Right channel")  # Right channel
        plt.title("Waveform - Stereo")
    
    # Add labels and legend
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show(block=False)  # Non-blocking mode
    plt.pause(0.001)

def display_frequency_response(file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Compute Short-Time Fourier Transform (STFT)
    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    time = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr)
    frequency = librosa.fft_frequencies(sr=sr)

    # Plot the frequency response (spectrogram)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(librosa.amplitude_to_db(magnitude, ref=np.max),
                             sr=sr, x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Frequency Response (Magnitude vs. Time)")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.tight_layout()
    plt.show()

# Function to process audio file and calculate RT60 for low, mid, and high frequency ranges
def process_Audio(file_path):
    global plots_Visible, time_curves, decay_curves, rt60_low, rt60_mid, rt60_high

    # Load the audio file and convert to mono if necessary
    audio, sr = librosa.load(file_path, sr=None, mono=False)
    if len(audio.shape) > 1:  # Check for multi-channel audio
        audio = np.mean(audio, axis=0) # Average the channels for RT60 calculation

    # Calculate and display the duration of the audio in seconds
    duration = len(audio) / sr  # Duration in seconds
    # Compute resonant frequency
    resonant_freq = compute_resonant_frequency(audio, sr)

    duration_label.config(text=f"Duration: {duration:.2f} seconds")
    resonant_freq_label.config(text=f"Resonant Frequency: {resonant_freq:.2f} Hz")

    # Display the waveform of the audio file
    display_waveform(file_path)

    # Calculate RT60 for low, mid, and high frequency ranges
    time_low, decay_low, rt60_low = compute_rt60(audio, sr, low_freq=20, high_freq=200)
    time_mid, decay_mid, rt60_mid = compute_rt60(audio, sr, low_freq=200, high_freq=2000)
    time_high, decay_high, rt60_high = compute_rt60(audio, sr, low_freq=2000, high_freq=20000)

    # Store time and decay data globally
    time_curves['low'] = time_low
    decay_curves['low'] = decay_low
    time_curves['mid'] = time_mid
    decay_curves['mid'] = decay_mid
    time_curves['high'] = time_high
    decay_curves['high'] = decay_high


    # If plots are visible, update them
    if plots_Visible:
        update_Plots(rt60_low, rt60_mid, rt60_high)
    return rt60_low, rt60_mid, rt60_high

def compute_resonant_frequency(audio_signal, sr):
    # Use Welch's method 
    freqs, psd = welch(audio_signal, fs=sr)
    
    # Find the frequency with the maximum power
    max_index = np.argmax(psd)
    dominant_freq = freqs[max_index]
    
    return dominant_freq

def compute_rt60(audio_signal, sr, low_freq, high_freq):
    # Apply bandpass filter
    nyquist = sr / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    
    filtered_signal = signal.filtfilt(b, a, audio_signal)

    # Compute the energy decay curve (EDC)
    envelope = np.abs(filtered_signal) ** 2
    
    decay = 10 * np.log10(envelope / np.max(envelope))

    # Finds time when the signal decays by 60 dB (RT60).
    time = np.arange(len(decay)) / sr
    rt60_index = np.where(decay >= -60)[0]
    rt60_time = time[rt60_index[0]] if len(rt60_index) > 0 else time[-1]

    return time, decay, rt60_time

# Function to update RT60 plots
def update_Plots(rt60_low, rt60_mid, rt60_high):
    global axes, current_plot_type, time_curves, decay_curves



    if np.isnan(rt60_low) or np.isnan(rt60_mid) or np.isnan(rt60_high):
            print("RT60 values are invalid. Cannot update plots.")
            axes.text(0.5, 0.5, "RT60 values not available.", fontsize=14, ha='center', va='center')
            plt.draw()
            return


    axes.clear()

    # Check if decay curves are valid before plotting
    if time_curves.get('low') is None or decay_curves.get('low') is None:
        print("No decay data available for low frequencies.")
    if time_curves.get('mid') is None or decay_curves.get('mid') is None:
        print("No decay data available for mid frequencies.")
    if time_curves.get('high') is None or decay_curves.get('high') is None:
        print("No decay data available for high frequencies.")

    if current_plot_type == 'all':
        # Plot all decay curves
        if time_curves['low'] is not None and decay_curves['low'] is not None:
            axes.plot(time_curves['low'], decay_curves['low'], label='Low (20-200 Hz)', color='blue')
        if time_curves['mid'] is not None and decay_curves['mid'] is not None:
            axes.plot(time_curves['mid'], decay_curves['mid'], label='Mid (200-2000 Hz)', color='green')
        if time_curves['high'] is not None and decay_curves['high'] is not None:
            axes.plot(time_curves['high'], decay_curves['high'], label='High (2000-20000 Hz)', color='red')

    elif current_plot_type == 'low':
        if time_curves['low'] is not None and decay_curves['low'] is not None:
            axes.plot(time_curves['low'], decay_curves['low'], label='Low (20-200 Hz)', color='blue')

    elif current_plot_type == 'mid':
        if time_curves['mid'] is not None and decay_curves['mid'] is not None:
            axes.plot(time_curves['mid'], decay_curves['mid'], label='Mid (200-2000 Hz)', color='green')

    elif current_plot_type == 'high':
        if time_curves['high'] is not None and decay_curves['high'] is not None:
            axes.plot(time_curves['high'], decay_curves['high'], label='High (2000-20000 Hz)', color='red')

    # Add labels, legend, and title
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Decay (dB)")
    axes.set_title("RT60 Decay Curves")
    axes.set_ylim(-100, 0)  # Set the y-axis limits to negative dB values
    axes.legend()
    plt.draw()
    plt.pause(0.001)

# Function to toggle the visibility of RT60 plots
def toggle_plots():
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

# Function to show the combined plots
def show_combined_plots():
    global current_plot_type, rt60_low, rt60_mid, rt60_high

    # Set plot type to 'all' and update the plots
    current_plot_type = 'all'
    update_Plots(rt60_low, rt60_mid, rt60_high)

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
        file_label.config(text=file_Path)
        
        # Process the file (convert to WAV if necessary)
        if not file_Path.endswith(".wav"):
            file_Path = convert_to_wav(file_Path)  # Assuming you implemented the conversion logic
        rt60_low, rt60_mid, rt60_high = process_Audio(file_Path)
        report = RT60_Report(rt60_low, rt60_mid, rt60_high)
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

# Resonant Frequency
resonant_freq_label = tk.Label(root, text="Resonant Frequency: 0 Hz", font=("Arial", 12))
resonant_freq_label.pack()


# Create a button for displaying the Frequency Response
freq_response_button = tk.Button(root, text="Show Frequency Response", command=lambda: display_frequency_response(file_label['text']))
freq_response_button.pack()

# Combined Plots Button
combined_button = tk.Button(root, text="Show Combined Plots", command=lambda: show_combined_plots())
combined_button.pack()

# Start the GUI main loop
root.mainloop()