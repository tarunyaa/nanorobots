#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 02:37:05 2022

@author: Taru
"""

# Imports
import cProfile
import pstats
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def gerchbergSaxtonAlgorithm(maxNoOfIterations): 
    
    # Opening the image and resizing to SLM dimensions
    
    old_im = Image.open('dot1.png')
    w = 1080
    h = 1920

    # Converting image to numpy array

    img = np.array(old_im)
    
    # Initialising variables
    
    targetPhase = 2 * np.pi * np.random.rand(w, h) - np.pi # Phase guess 
    targetAmplitude = np.sqrt(img)
    targetAmplitude = targetAmplitude[:, :, 0]
    sourceAmplitude = np.ones((w, h))    
    sourcePhase = np.ones((w, h))    
    targetField = targetAmplitude * np.exp(targetPhase * complex(0, 1)) 
    targetFieldOriginal = targetField    
    k = 1
    
    # Gerchberg-Saxton Algorithm
    while (k == 1) or ((k < maxNoOfIterations)):
        
        sourceField = np.fft.fft2(targetField)  #fft2 is the 2D fourier 
        sourcePhase = np.angle(sourceField)
        
        sourceField = sourceAmplitude * np.exp(complex(0, 1) * sourcePhase)
        targetField = np.fft.ifft2(sourceField) #ifft is inverse fourier 
        targetPhase = np.angle(targetField)
                
        targetField = targetAmplitude * np.exp(complex(0, 1) * targetPhase)
        
        k += 1
    return sourceField

maxNoOfIterations = 1

pr = cProfile.Profile()
pr.enable()

# Printing Source Field
sourceField = gerchbergSaxtonAlgorithm(maxNoOfIterations)

# Printing Source Phase
sourcePhase = np.angle(sourceField)

# Saving Source phase image
plt.imsave('SP.png',sourcePhase, cmap = 'gray')
    
pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('cProfileTest3.txt', 'w+') as f:
    f.write(s.getvalue())



