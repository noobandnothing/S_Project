#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 01:18:16 2025

@author: noob
"""
from AnomalyFixer import  AnomalyFixer


obj = AnomalyFixer()
obj.process_all_png_in_dir(
    '/home/noob/koty/new_before_last/work/data/model_data-after-fix/')

