import numpy as np
import matplotlib.pyplot as plt

def simulate_digitization(analog_signal, sampling_rate, quantization_levels):
    """
    analog_signal: fungsi kontinu f(x)
    sampling_rate: jumlah sampel
    quantization_levels: jumlah level kuantisasi
    """
    
    # Sinyal kontinu
    x_cont = np.linspace(0, 1, 1000)
    y_cont = analog_signal(x_cont)
    
    # 1. Sampling
    x_sample = np.linspace(0, 1, sampling_rate)
    y_sample = analog_signal(x_sample)
    
    # 2. Quantization
    y_min = np.min(y_sample)
    y_max = np.max(y_sample)
    q_step = (y_max - y_min) / quantization_levels
    y_quant = np.round((y_sample - y_min) / q_step) * q_step + y_min
    
    # 3. Visualisasi
    plt.figure()
    plt.plot(x_cont, y_cont, label="Sinyal Analog")
    plt.stem(x_sample, y_quant, linefmt='r-', markerfmt='ro', basefmt=' ')
    plt.title("Simulasi Sampling & Quantization")
    plt.legend()
    plt.show()


# Contoh penggunaan
simulate_digitization(
    analog_signal=lambda x: np.sin(2*np.pi*5*x),
    sampling_rate=20,
    quantization_levels=8
)