#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: COM2 Simulador Headless GNU Radio
# Author: Proyecto COM2
# Description: Simulador transmisor y canal sintetico sin hardware para COM2
# GNU Radio version: 3.10.9.2

from gnuradio import blocks
import numpy
from gnuradio import channels
from gnuradio.filter import firdes
from gnuradio import digital
from gnuradio import filter
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio.filter import firdes; import numpy as np




class flowgraph_headless_sim(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "COM2 Simulador Headless GNU Radio", catch_exceptions=True)

        ##################################################
        # Variables
        ##################################################
        self.sps = sps = 8
        self.samp_rate = samp_rate = 8000
        self.baud_rate = baud_rate = 1000
        self.alpha = alpha = 0.35
        self.rrc_taps = rrc_taps = firdes.root_raised_cosine(sps, samp_rate, baud_rate, alpha, 81)
        self.qpsk_const = qpsk_const = [(1.0+1.0j)*0.70710678, (-1.0+1.0j)*0.70710678, (-1.0-1.0j)*0.70710678, (1.0-1.0j)*0.70710678]
        self.out_file = out_file = "grc_sim_output.c64"
        self.noise_volt = noise_volt = 0.02
        self.nitems = nitems = 4096
        self.cfo_norm = cfo_norm = 0.0

        ##################################################
        # Blocks
        ##################################################

        self.interp_fir_filter_xxx_0 = filter.interp_fir_filter_ccc(sps, rrc_taps)
        self.interp_fir_filter_xxx_0.declare_sample_delay(0)
        self.digital_chunks_to_symbols_xx_0 = digital.chunks_to_symbols_bc(qpsk_const, 1)
        self.channels_channel_model_0 = channels.channel_model(
            noise_voltage=noise_volt,
            frequency_offset=cfo_norm,
            epsilon=1.0,
            taps=[1.0],
            noise_seed=42,
            block_tags=False)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, nitems)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, out_file, False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.analog_random_source_x_0 = blocks.vector_source_b(list(map(int, numpy.random.randint(0, 4, 20000))), True)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_random_source_x_0, 0), (self.digital_chunks_to_symbols_xx_0, 0))
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.channels_channel_model_0, 0), (self.blocks_head_0, 0))
        self.connect((self.digital_chunks_to_symbols_xx_0, 0), (self.interp_fir_filter_xxx_0, 0))
        self.connect((self.interp_fir_filter_xxx_0, 0), (self.channels_channel_model_0, 0))


    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.set_rrc_taps(firdes.root_raised_cosine(self.sps, self.samp_rate, self.baud_rate, self.alpha, 81))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_rrc_taps(firdes.root_raised_cosine(self.sps, self.samp_rate, self.baud_rate, self.alpha, 81))

    def get_baud_rate(self):
        return self.baud_rate

    def set_baud_rate(self, baud_rate):
        self.baud_rate = baud_rate
        self.set_rrc_taps(firdes.root_raised_cosine(self.sps, self.samp_rate, self.baud_rate, self.alpha, 81))

    def get_alpha(self):
        return self.alpha

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.set_rrc_taps(firdes.root_raised_cosine(self.sps, self.samp_rate, self.baud_rate, self.alpha, 81))

    def get_rrc_taps(self):
        return self.rrc_taps

    def set_rrc_taps(self, rrc_taps):
        self.rrc_taps = rrc_taps
        self.interp_fir_filter_xxx_0.set_taps(self.rrc_taps)

    def get_qpsk_const(self):
        return self.qpsk_const

    def set_qpsk_const(self, qpsk_const):
        self.qpsk_const = qpsk_const
        self.digital_chunks_to_symbols_xx_0.set_symbol_table(self.qpsk_const)

    def get_out_file(self):
        return self.out_file

    def set_out_file(self, out_file):
        self.out_file = out_file
        self.blocks_file_sink_0.open(self.out_file)

    def get_noise_volt(self):
        return self.noise_volt

    def set_noise_volt(self, noise_volt):
        self.noise_volt = noise_volt
        self.channels_channel_model_0.set_noise_voltage(self.noise_volt)

    def get_nitems(self):
        return self.nitems

    def set_nitems(self, nitems):
        self.nitems = nitems
        self.blocks_head_0.set_length(self.nitems)

    def get_cfo_norm(self):
        return self.cfo_norm

    def set_cfo_norm(self, cfo_norm):
        self.cfo_norm = cfo_norm
        self.channels_channel_model_0.set_frequency_offset(self.cfo_norm)




def main(top_block_cls=flowgraph_headless_sim, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    tb.wait()


if __name__ == '__main__':
    main()
