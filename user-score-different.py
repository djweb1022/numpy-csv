# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/8 22:57'

import pandas as pd
import numpy as np

userscore = pd.read_csv('user-score.csv')

AAA_2013J = userscore[(userscore['code_module'] == 'AAA') & (userscore['code_presentation'] == '2013J')]
AAA_2014J = userscore[(userscore['code_module'] == 'AAA') & (userscore['code_presentation'] == '2014J')]
BBB_2013J = userscore[(userscore['code_module'] == 'BBB') & (userscore['code_presentation'] == '2013J')]
BBB_2014J = userscore[(userscore['code_module'] == 'BBB') & (userscore['code_presentation'] == '2014J')]
BBB_2013B = userscore[(userscore['code_module'] == 'BBB') & (userscore['code_presentation'] == '2013B')]
BBB_2014B = userscore[(userscore['code_module'] == 'BBB') & (userscore['code_presentation'] == '2014B')]
CCC_2014J = userscore[(userscore['code_module'] == 'CCC') & (userscore['code_presentation'] == '2014J')]
CCC_2014B = userscore[(userscore['code_module'] == 'CCC') & (userscore['code_presentation'] == '2014B')]
DDD_2013J = userscore[(userscore['code_module'] == 'DDD') & (userscore['code_presentation'] == '2013J')]
DDD_2014J = userscore[(userscore['code_module'] == 'DDD') & (userscore['code_presentation'] == '2014J')]
DDD_2013B = userscore[(userscore['code_module'] == 'DDD') & (userscore['code_presentation'] == '2013B')]
DDD_2014B = userscore[(userscore['code_module'] == 'DDD') & (userscore['code_presentation'] == '2014B')]
EEE_2013J = userscore[(userscore['code_module'] == 'EEE') & (userscore['code_presentation'] == '2013J')]
EEE_2014J = userscore[(userscore['code_module'] == 'EEE') & (userscore['code_presentation'] == '2014J')]
EEE_2014B = userscore[(userscore['code_module'] == 'EEE') & (userscore['code_presentation'] == '2014B')]
FFF_2013J = userscore[(userscore['code_module'] == 'FFF') & (userscore['code_presentation'] == '2013J')]
FFF_2014J = userscore[(userscore['code_module'] == 'FFF') & (userscore['code_presentation'] == '2014J')]
FFF_2013B = userscore[(userscore['code_module'] == 'FFF') & (userscore['code_presentation'] == '2013B')]
FFF_2014B = userscore[(userscore['code_module'] == 'FFF') & (userscore['code_presentation'] == '2014B')]
GGG_2013J = userscore[(userscore['code_module'] == 'GGG') & (userscore['code_presentation'] == '2013J')]
GGG_2014J = userscore[(userscore['code_module'] == 'GGG') & (userscore['code_presentation'] == '2014J')]
GGG_2014B = userscore[(userscore['code_module'] == 'GGG') & (userscore['code_presentation'] == '2014B')]

shape_sum = AAA_2013J.shape[0] + AAA_2014J.shape[0] + BBB_2013J.shape[0] + BBB_2014J.shape[0] + BBB_2013B.shape[0] + \
            CCC_2014J.shape[0] + CCC_2014B.shape[0] + DDD_2013J.shape[0] + DDD_2014J.shape[0] + DDD_2013B.shape[0] + \
            DDD_2014B.shape[0] + EEE_2013J.shape[0] + EEE_2014J.shape[0] + EEE_2014B.shape[0] + FFF_2013J.shape[0] + \
            FFF_2014J.shape[0] + FFF_2013B.shape[0] + FFF_2014B.shape[0] + GGG_2013J.shape[0] + GGG_2014J.shape[0] + \
            GGG_2014B.shape[0] + BBB_2014B.shape[0]

print(userscore)
print(shape_sum)
# 171503

AAA_2013J.to_csv('different_userscore/AAA_2013J_user-score.csv', index=False)
AAA_2014J.to_csv('different_userscore/AAA_2014J_user-score.csv', index=False)
BBB_2013J.to_csv('different_userscore/BBB_2013J_user-score.csv', index=False)
BBB_2014J.to_csv('different_userscore/BBB_2014J_user-score.csv', index=False)
BBB_2013B.to_csv('different_userscore/BBB_2013B_user-score.csv', index=False)
BBB_2014B.to_csv('different_userscore/BBB_2014B_user-score.csv', index=False)
CCC_2014J.to_csv('different_userscore/CCC_2014J_user-score.csv', index=False)
CCC_2014B.to_csv('different_userscore/CCC_2014B_user-score.csv', index=False)
DDD_2013J.to_csv('different_userscore/DDD_2013J_user-score.csv', index=False)
DDD_2014J.to_csv('different_userscore/DDD_2014J_user-score.csv', index=False)
DDD_2013B.to_csv('different_userscore/DDD_2013B_user-score.csv', index=False)
DDD_2014B.to_csv('different_userscore/DDD_2014B_user-score.csv', index=False)
EEE_2013J.to_csv('different_userscore/EEE_2013J_user-score.csv', index=False)
EEE_2014J.to_csv('different_userscore/EEE_2014J_user-score.csv', index=False)
EEE_2014B.to_csv('different_userscore/EEE_2014B_user-score.csv', index=False)
FFF_2013J.to_csv('different_userscore/FFF_2013J_user-score.csv', index=False)
FFF_2014J.to_csv('different_userscore/FFF_2014J_user-score.csv', index=False)
FFF_2013B.to_csv('different_userscore/FFF_2013B_user-score.csv', index=False)
FFF_2014B.to_csv('different_userscore/FFF_2014B_user-score.csv', index=False)
GGG_2013J.to_csv('different_userscore/GGG_2013J_user-score.csv', index=False)
GGG_2014J.to_csv('different_userscore/GGG_2014J_user-score.csv', index=False)
GGG_2014B.to_csv('different_userscore/GGG_2014B_user-score.csv', index=False)