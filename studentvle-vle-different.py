# -*- coding:utf-8 -*-
__author__ = 'yfj'
__date__ = '2019/5/8 22:19'

import pandas as pd
import numpy as np

studentvle = pd.read_csv('studentvle-vle.csv')

AAA_2013J = studentvle[(studentvle['code_module'] == 'AAA') & (studentvle['code_presentation'] == '2013J')]
AAA_2014J = studentvle[(studentvle['code_module'] == 'AAA') & (studentvle['code_presentation'] == '2014J')]
BBB_2013J = studentvle[(studentvle['code_module'] == 'BBB') & (studentvle['code_presentation'] == '2013J')]
BBB_2014J = studentvle[(studentvle['code_module'] == 'BBB') & (studentvle['code_presentation'] == '2014J')]
BBB_2013B = studentvle[(studentvle['code_module'] == 'BBB') & (studentvle['code_presentation'] == '2013B')]
BBB_2014B = studentvle[(studentvle['code_module'] == 'BBB') & (studentvle['code_presentation'] == '2014B')]
CCC_2014J = studentvle[(studentvle['code_module'] == 'CCC') & (studentvle['code_presentation'] == '2014J')]
CCC_2014B = studentvle[(studentvle['code_module'] == 'CCC') & (studentvle['code_presentation'] == '2014B')]
DDD_2013J = studentvle[(studentvle['code_module'] == 'DDD') & (studentvle['code_presentation'] == '2013J')]
DDD_2014J = studentvle[(studentvle['code_module'] == 'DDD') & (studentvle['code_presentation'] == '2014J')]
DDD_2013B = studentvle[(studentvle['code_module'] == 'DDD') & (studentvle['code_presentation'] == '2013B')]
DDD_2014B = studentvle[(studentvle['code_module'] == 'DDD') & (studentvle['code_presentation'] == '2014B')]
EEE_2013J = studentvle[(studentvle['code_module'] == 'EEE') & (studentvle['code_presentation'] == '2013J')]
EEE_2014J = studentvle[(studentvle['code_module'] == 'EEE') & (studentvle['code_presentation'] == '2014J')]
EEE_2014B = studentvle[(studentvle['code_module'] == 'EEE') & (studentvle['code_presentation'] == '2014B')]
FFF_2013J = studentvle[(studentvle['code_module'] == 'FFF') & (studentvle['code_presentation'] == '2013J')]
FFF_2014J = studentvle[(studentvle['code_module'] == 'FFF') & (studentvle['code_presentation'] == '2014J')]
FFF_2013B = studentvle[(studentvle['code_module'] == 'FFF') & (studentvle['code_presentation'] == '2013B')]
FFF_2014B = studentvle[(studentvle['code_module'] == 'FFF') & (studentvle['code_presentation'] == '2014B')]
GGG_2013J = studentvle[(studentvle['code_module'] == 'GGG') & (studentvle['code_presentation'] == '2013J')]
GGG_2014J = studentvle[(studentvle['code_module'] == 'GGG') & (studentvle['code_presentation'] == '2014J')]
GGG_2014B = studentvle[(studentvle['code_module'] == 'GGG') & (studentvle['code_presentation'] == '2014B')]

shape_sum = AAA_2013J.shape[0] + AAA_2014J.shape[0] + BBB_2013J.shape[0] + BBB_2014J.shape[0] + BBB_2013B.shape[0] + \
            CCC_2014J.shape[0] + CCC_2014B.shape[0] + DDD_2013J.shape[0] + DDD_2014J.shape[0] + DDD_2013B.shape[0] + \
            DDD_2014B.shape[0] + EEE_2013J.shape[0] + EEE_2014J.shape[0] + EEE_2014B.shape[0] + FFF_2013J.shape[0] + \
            FFF_2014J.shape[0] + FFF_2013B.shape[0] + FFF_2014B.shape[0] + GGG_2013J.shape[0] + GGG_2014J.shape[0] + \
            GGG_2014B.shape[0] + BBB_2014B.shape[0]

print(shape_sum)

AAA_2013J.to_csv('different_courses/AAA_2013J_studentvle-vle.csv', index=False)
AAA_2014J.to_csv('different_courses/AAA_2014J_studentvle-vle.csv', index=False)
BBB_2013J.to_csv('different_courses/BBB_2013J_studentvle-vle.csv', index=False)
BBB_2014J.to_csv('different_courses/BBB_2014J_studentvle-vle.csv', index=False)
BBB_2013B.to_csv('different_courses/BBB_2013B_studentvle-vle.csv', index=False)
BBB_2014B.to_csv('different_courses/BBB_2014B_studentvle-vle.csv', index=False)
CCC_2014J.to_csv('different_courses/CCC_2014J_studentvle-vle.csv', index=False)
CCC_2014B.to_csv('different_courses/CCC_2014B_studentvle-vle.csv', index=False)
DDD_2013J.to_csv('different_courses/DDD_2013J_studentvle-vle.csv', index=False)
DDD_2014J.to_csv('different_courses/DDD_2014J_studentvle-vle.csv', index=False)
DDD_2013B.to_csv('different_courses/DDD_2013B_studentvle-vle.csv', index=False)
DDD_2014B.to_csv('different_courses/DDD_2014B_studentvle-vle.csv', index=False)
EEE_2013J.to_csv('different_courses/EEE_2013J_studentvle-vle.csv', index=False)
EEE_2014J.to_csv('different_courses/EEE_2014J_studentvle-vle.csv', index=False)
EEE_2014B.to_csv('different_courses/EEE_2014B_studentvle-vle.csv', index=False)
FFF_2013J.to_csv('different_courses/FFF_2013J_studentvle-vle.csv', index=False)
FFF_2014J.to_csv('different_courses/FFF_2014J_studentvle-vle.csv', index=False)
FFF_2013B.to_csv('different_courses/FFF_2013B_studentvle-vle.csv', index=False)
FFF_2014B.to_csv('different_courses/FFF_2014B_studentvle-vle.csv', index=False)
GGG_2013J.to_csv('different_courses/GGG_2013J_studentvle-vle.csv', index=False)
GGG_2014J.to_csv('different_courses/GGG_2014J_studentvle-vle.csv', index=False)
GGG_2014B.to_csv('different_courses/GGG_2014B_studentvle-vle.csv', index=False)




# [10655280 rows x 7 columns]