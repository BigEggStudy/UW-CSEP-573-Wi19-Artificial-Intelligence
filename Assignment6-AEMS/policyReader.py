#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UW, CSEP 573, Win19
"""

import xml.etree.ElementTree as ET
import numpy as np

class PolicyReader:
    def __init__(self, pomdp, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        avec = list(root)[0]
        alphas = list(avec)
        self.action_nums = []
        val_arrs = []
        for alpha in alphas:
            self.action_nums.append(int(alpha.attrib['action']))
            vals = []
            for val in alpha.text.split():
                vals.append(float(val))
            val_arrs.append(vals)
        self.pMatrix = np.array(val_arrs)
        self.pomdp = pomdp

    def chooseAction(self, cur_belief):
        res = self.pMatrix.dot(cur_belief.flatten())
        best_action = self.action_nums[res.argmax()]
        return best_action
    
    def getValue(self, cur_belief):
        res = self.pMatrix.dot(cur_belief.flatten())
        highest_expected_reward = res.max()
        return highest_expected_reward
