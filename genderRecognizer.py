from __future__ import division
from pylab import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy import *
from scipy import signal
import scipy.io.wavfile
import time


class Plotter:

    def __init__(self, size_y, size_x=1):
        self.fig = plt.figure(figsize=(20, 80), dpi=80)
        self.size_x = size_x
        self.size_y = size_y
        self.plotNumber = 0

    def draw_plot(self, x_values, y_values):
        self.plotNumber += 1
        ax = self.fig.add_subplot(f"{self.size_y}{self.size_x}{self.plotNumber}")
        xlabel("Frequency[Hz]", fontsize=18, color="r")
        ylabel("Amplitude[unit]", fontsize=18, color="r")
        ax.set_xlim(-5, 1000)
        ax.plot(x_values, y_values, '-*')


class GenderRecognizer:

    def __init__(self, plotter=None):
        self.plotter = plotter

    def to_one_canal(self, in_signal):
        signal0 = in_signal
        if type(signal0[0]) is np.ndarray:
            signal0 = [s[0] for s in signal0]       # Take only one canal
        return signal0

    def use_kaiser_mask(self, in_signal):
        return in_signal * kaiser(len(in_signal), 50)  # kaiser mask

    def calc_abs_fft(self, in_signal, num_of_probes):
        signal1 = fft(in_signal)                      # signal in domain of frequency
        signal1 = abs(signal1)                      # module of signal
        signal1 = signal1 / (num_of_probes / 2)
        signal1[0] /= 2
        return signal1

    def calc_freqs_of_fft(self, num_of_probes, probes_per_second):
        return [(i / num_of_probes) * probes_per_second for i in range(num_of_probes)]

    def recognize_gender(self, freq):
        if freq > 171:
            return "K"
        else:
            return "M"

    def getGender(self, file_path):
        probes_per_second, signal0 = scipy.io.wavfile.read(file_path)

        signal0 = self.to_one_canal(signal0)
        signal0 = self.use_kaiser_mask(signal0)
        num_of_probes = len(signal0)

        df = pd.DataFrame(
            {'frequency': self.calc_freqs_of_fft(num_of_probes, probes_per_second),
             'value': self.calc_abs_fft(signal0, num_of_probes)
             })

        df[df['frequency'] < 70] = 0  # lower bound of frequencies
        df[df['frequency'] > 20000] = 0  # upper bound of frequencies

        original_fft = df.copy()

        if self.plotter is not None:
            self.plotter.draw_plot(original_fft['frequency'], original_fft['value'])

        for p in arange(2, 5):
            d = signal.decimate(original_fft['value'], p)
            df['value'][:len(d)] *= d

            if self.plotter is not None:
                self.plotter.draw_plot(df['frequency'][:len(d):10], d[::10])

        df_sorted = df.sort_values(by='value', ascending=False)

        if self.plotter is not None:
            self.plotter.draw_plot(df['frequency'], df['value'])
            print(df_sorted.head(15))

        dominating_freq = df_sorted.head(1).iloc[0]['frequency']
        return self.recognize_gender(dominating_freq)


if __name__ == '__main__':

    file = "Interviewer.wav"
    if len(sys.argv) > 1:
        file = sys.argv[1]

    gender_recognizer = GenderRecognizer()
    if len(sys.argv) > 2 and sys.argv[2] == str(1):
        plotter = Plotter(8)
        gender_recognizer = GenderRecognizer(plotter)

    recognizedGender = gender_recognizer.getGender(file)
    print(recognizedGender)

    if len(sys.argv) > 2 and sys.argv[2] == str(1):
        plt.rcParams.update(
            {'font.size': 40, 'font.family': 'Times New Roman', 'font.weight': 'light', 'xtick.color': 'orange',
             'ytick.color': 'orange', 'text.color': 'blue'})

        # plt.savefig('plot.png')
        plt.show()
