# MIT License
#
# Copyright (c) 2021 Madalin Cherciu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#!/bin/python
import os
import sys, gc
import time
import numpy as np
import pylandau
import array as arr
# for gaussian fit
import math
import scipy.stats as stats
#from scipy.stats import poisson
# to work within the files
from scipy import special
from itertools import islice
# to plot
import matplotlib.pyplot as plt
#to format axex
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MultipleLocator, ScalarFormatter
# calculate integrals
import scipy.integrate as integrate
FIG1="ON"
FIG2="OFF"

def main():
     filename1 = open("file1.txt","r")
     outfile = open("file1_area_channels.txt","w+")
     cntval = 0
     cntpuls = 0
     cnt_bad_pulses = 0
     cnt_valid_pulse = 0
     cnt_non_pulses = 0
     x = np.linspace(0,60,60)
     y = arr.array('I', [])
     area = arr.array('I',[])

     if (FIG1 == "ON"):
         fig, ax = plt.subplots()
         ax.xaxis.set_minor_locator(MultipleLocator(1))
         ax.xaxis.set_minor_formatter(ScalarFormatter())
#         ax.set_xlim([0, 8000])

     if (FIG2 == "ON"):
         fig1, ax1 = plt.subplots()
         ax1.xaxis.set_major_locator(MultipleLocator(100))
         ax1.xaxis.set_minor_locator(MultipleLocator(50))
         ax1.set_xlim([700, 1100])
         ax1.minorticks_on()
         ax1.set_xlabel('Channel')
         ax1.set_ylabel('Counts')

     def gaussian(x, ylow, yhigh, mean, sigma):
        return ylow + yhigh * np.exp(-(x - mean)**2 / sigma**2) 

     def gaussian_fit(bin, high, gf_sigma, baseline):
        xm = [0]*6*gf_sigma
        ym = 0
        area_sum = 0
        mu = bin
#        variance = gf_sigma^2
        sigma = gf_sigma
        xmin = mu-2.35*sigma
        xmax = mu+2.35*sigma
        topend = high - baseline
        xm = np.linspace(xmin, xmax, 6*gf_sigma)
        ym = gaussian(xm, baseline, topend, mu, sigma)

        return xm,ym

     def pulse_height(ph_ymax):
        area_sum = ph_ymax
        return area_sum

     def fill_histo(fl_maxy):
         gf_format = float("{:.2f}".format(fl_maxy))
         area.append(int(gf_format))

     def chkpos_pulse(cp_y, cp_maxx, cp_maxy):
        if (cp_maxx >= 8 and cp_maxx <= 45):
           flag = True
        else:
           cp_diff = cp_y[cp_maxx] - cp_y[cp_maxx-1]
           if(cp_diff > 100):
              cp_y[cp_maxx] = cp_y[cp_maxx-1]
              cp_maxElement = np.amax(cp_y)
              cp_maxBin = find_xmax(cp_y, cp_maxElement)
              if(cp_maxBin >=8 and cp_maxBin <=45):
                  flag = True
                  cp_maxx = cp_maxBin
                  cp_maxy = cp_maxElement
              else:
                  flag = False
           else:
               flag = False
        return flag, cp_maxx, cp_maxy

     def find_xmax(fm_y, fm_ymax):
        fm_xmin = np.argmax(fm_y)
        return int(fm_xmin)


     def noise_loop(nl_y, nl_xmax):
         flag = False
         yprec = 0
         cnt1 = 0
         cnt2 = 0
         cntoff = 0
         nl_y_reco = [0]*25
         if(nl_xmax > 0):
            for i in reversed(range(nl_xmax)):
               i = nl_xmax-9-cnt1-cntoff
               if (flag == True):
                  flag = False
               if (cnt1 == 0):
                     nl_y_reco[cnt1] = nl_y[i]
                     cnt1 += 1
                     yprec = nl_y[i]
               elif(cnt1>0 and cnt1 <25):
                  diff = nl_y[i] - yprec
                  if( abs(diff) <= 100 ):
                     yprec = nl_y[i]
                     nl_y_reco[cnt1] = yprec
                     cnt1 += 1
                  else:
                     flag = True
                     cntoff += 1
         if(cnt1 > 0):
               nl_y_mean = sum(nl_y_reco)/cnt1
         return nl_y_mean

     def find_baseline(bl_y, bl_xmax, bl_ymax):
        bl_ymean = 0
        bl_ymean = noise_loop(bl_y, bl_xmax)
        return bl_ymean

     def find_sigma(fs_y, fs_xmed, fs_ymax, fs_bln):
        fs_mem = 0
        fs_xmin = 0
        fs_xmax = 0
        fs_ymed = fs_bln + int(round((fs_ymax - fs_bln) / 2))
        for ifs in range(fs_xmed,0,-1):
           fs_mem = fs_y[ifs]
           if(fs_mem < fs_ymed):
              fs_xmin = ifs
 
        for jfs in range(fs_xmed+1, 60,1):
           fs_mem = fs_y[jfs]
           if(fs_mem < fs_ymed):
              fs_xmax = jfs
        fs_sigma = int( math.ceil( (fs_xmax-fs_xmin)/2.35) )
        return fs_sigma

     def try_recovery(tr_y, tr_maxx, tr_maxy):
         tr_y_reco = [0]*60
         tr_cnt = 0
         half_plus = 0
         half_minus = 0
         for i in range(30):
            tr_cnt += 1
            
            if(i < 30):
               half_plus =  i + 30
               half_minus = 30 - tr_cnt
               if (i == 0):
                  if(abs(tr_y[i+1] - tr_y[i]) > 50):
                     tr_y_reco[half_plus] = tr_y[i+1] + (tr_y[i+2] - tr_y[i+1])/2
                  else:
                     tr_y_reco[half_plus] = tr_y[i]   
                  if(abs(tr_y[60-tr_cnt] - tr_y[60-tr_cnt-1]) > 50):
                     tr_y_reco[half_minus] = tr_y[60-tr_cnt-1] - (tr_y[60-tr_cnt-2] - tr_y[60-tr_cnt-1])/2
                  else:
                     tr_y_reco[half_plus] = tr_y[i]
                     tr_y_reco[half_minus] = tr_y[60-tr_cnt]   
               else:
                  tr_y_reco[half_plus] = tr_y[i]
                  tr_y_reco[half_minus] = tr_y[60-tr_cnt]
         tr_cnt = 0
         
         return tr_y_reco

     def plot_pulse(pp_x, pp_y,pp_no):
         ax.set_xlabel('Channel')
         ax.set_ylabel('ADC value')
# just for interactive display
         ax.minorticks_on()
         fig.suptitle(pp_no)
         ax.plot(pp_x,pp_y,'b:o')
         plt.draw()
         plt.waitforbuttonpress()     # Display plot and wait for user input.
         ax.clear()   # widthout this will accumulate pulses in different color


     for line1 in islice(filename1, 1):
        bflag = False
        maxx = 0
        maxy = 0

        for pulses in line1.split('&'):
            for values in pulses.split(','):
                if values != '':
                    cntval+=1
                    y.append(int(values))
            if (cntval == 60 and len(y) == 60):
               maxElement = np.amax(y)
               maxBin = find_xmax(y, maxElement)
               bflag, maxx, maxy = chkpos_pulse(y, maxBin, maxElement)
               if ( bflag and cntpuls >= 1):
                  blline = find_baseline(y, maxx, maxy)
                  step = maxy - blline
                  if blline <= 6000 and step>100:
                     cnt_valid_pulse += 1
                     if(FIG1 == "ON"):
#just for interactive display ... valid pulses
                        ybl = [blline]*60
                        ax.plot(x,ybl,'r--')
                        plot_pulse(x,y,cnt_valid_pulse)
                     if(FIG2 == "ON"):
                        sumarea = pulse_height(maxy)
                        if (sumarea > 0):
                           area.append(int(sumarea))
                  else:
                     cnt_bad_pulses += 1 
                  new_y = try_recovery(y, maxx, maxy)
                  maxElement = np.amax(new_y)
                  maxBin = find_xmax(new_y, maxElement)
                  if (maxBin > 8 and maxBin < 45):
                     blline = find_baseline(new_y, maxBin, maxElement)
                     step = maxElement - blline
                     if blline <= 600 and step>100:
                        cnt_valid_pulse += 1
                        if(FIG1 == "ON"):
                           ybl = [blline]*60
                           ax.plot(x,ybl,'r--')
                           plot_pulse(x,new_y,cnt_valid_pulse)
                        if(FIG2 == "ON"):
                           sumarea = pulse_height(maxElement)
                           if (sumarea > 0):
                              area.append(int(sumarea))
                  else:
                     cnt_non_pulses += 1
                  
            if len(pulses) > 0:
                cntpuls += 1
            cntval = 0
            flag = False
            y = []
# this is for displaying puleses area plot
        if(FIG2 == "ON"):
            binwidth = 5
            ax1.hist(area, bins = np.arange(min(area), max(area) + binwidth, binwidth),color='g')
            ax1.text(870, 13000, u"\u0394E = 7%")
            gfx1, gfy1 = gaussian_fit(847, 1690, 23, 0)
            ax1.plot(gfx1, gfy1, color='r')
#            plt.yscale("log")
            plt.title(r"$\gamma$ spectrum of $CS^{137}$ source")
            plt.show()
#-------------------------------------------------------------------------------------------
        print("Total PULSES = ", cntpuls, "Valid pulses = ", cnt_valid_pulse, "Bad pulses = ", cnt_bad_pulses," Non pulse = ", cnt_non_pulses)
        for listitem in area:
           outfile.write(str(listitem) + "\n")
        outfile.close()

if __name__== "__main__":
  main()


