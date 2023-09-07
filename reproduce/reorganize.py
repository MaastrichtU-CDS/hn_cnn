# Reorganize the data in folders as expected by the code
# developed by Diamant et al.

import os
import shutil

ids = ['HN1004','HN1006','HN1022','HN1026','HN1029','HN1046','HN1047','HN1054','HN1057','HN1060','HN1062','HN1067','HN1074','HN1077','HN1079','HN1080','HN1081','HN1083','HN1088','HN1092','HN1095','HN1096','HN1102','HN1106','HN1117','HN1118','HN1123','HN1127','HN1135','HN1139','HN1146','HN1159','HN1170','HN1175','HN1180','HN1192','HN1197','HN1200','HN1201','HN1208','HN1215','HN1244','HN1259','HN1260','HN1263','HN1271','HN1280','HN1294','HN1305','HN1308','HN1310','HN1319','HN1323','HN1324','HN1327','HN1331','HN1339','HN1342','HN1344','HN1355','HN1356','HN1357','HN1363','HN1367','HN1368','HN1369','HN1371','HN1372','HN1395','HN1400','HN1412','HN1417','HN1429','HN1442','HN1444','HN1461','HN1465','HN1469','HN1483','HN1485','HN1486','HN1487','HN1488','HN1491','HN1500','HN1501','HN1502','HN1514','HN1517','HN1519','HN1524','HN1538','HN1549','HN1554','HN1555','HN1560','HN1562','HN1572','HN1600','HN1609','HN1610','HN1640','HN1648','HN1653','HN1667','HN1679','HN1697','HN1703','HN1719','HN1748','HN1760','HN1791','HN1792','HN1793','HN1805','HN1813','HN1815','HN1827','HN1838','HN1839','HN1851','HN1860','HN1869','HN1879','HN1892','HN1896','HN1900','HN1901','HN1910','HN1913','HN1922','HN1933','HN1950','HN1954','HN1968','HN1987','HN1998']

event = [0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,0,1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,1,1,0,0,1,0,1,0,0]

folder = 'TEST'
event = '0'

source_path = ''

path = f'{source_path}/data/original/maastro/'
destpath = f'{source_path}/data/IMAGES_LRF_CENTER/{folder}/{event}'
scans = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f not in ['.DS_Store']]

for event in ['0', '1']:
    for scan in scans:
        if scan.split('.')[0] in ids:
            if event[ids.index(scan.split('.')[0])] == int(event):
                shutil.copyfile(os.path.join(path, scan), os.path.join(destpath, scan))
