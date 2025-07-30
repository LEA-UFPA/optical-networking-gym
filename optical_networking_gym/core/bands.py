class Band:
    def __init__(self, name, freq_start, freq_end, num_slots, noise_figure, attenuation, input_power):
        self.name = name
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.num_slots = num_slots
        self.noise_figure = noise_figure
        self.attenuation = attenuation
        self.input_power = input_power

class BandS(Band):
    def __init__(self):
        super().__init__("S", 197.22, 205.30, 647, 7.0, 0.220, -0.38)

class BandC(Band):
    def __init__(self):
        super().__init__("C", 191.60, 195.90, 344, 5.5, 0.191, -3.66)

class BandL(Band):
    def __init__(self):
        super().__init__("L", 185.83, 190.90, 406, 6.0, 0.200, -2.78) 