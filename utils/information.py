bj_station_list = ['dongsi_aq','tiantan_aq','guanyuan_aq','wanshouxigong_aq','aotizhongxin_aq',
        'nongzhanguan_aq','wanliu_aq','beibuxinqu_aq','zhiwuyuan_aq','fengtaihuayuan_aq',
        'yungang_aq','gucheng_aq','fangshan_aq','daxing_aq','yizhuang_aq','tongzhou_aq',
        'shunyi_aq','pingchang_aq','mentougou_aq','pinggu_aq','huairou_aq','miyun_aq',
        'yanqin_aq','dingling_aq','badaling_aq','miyunshuiku_aq','donggaocun_aq',
        'yongledian_aq','yufa_aq','liulihe_aq','qianmen_aq','yongdingmennei_aq',
        'xizhimenbei_aq','nansanhuan_aq','dongsihuan_aq']        
bj_X_aq_list = ["PM2.5","PM10","O3","CO","SO2","NO2"]  
bj_y_aq_list = ["PM2.5","PM10","O3"]
bj_X_meo_list = ["temperature","pressure","humidity","direction","speed/kph"]

ld_station_list = ['BL0','CD1','CD9','GN0','GN3','GR4','GR9','HV1','KF1','LW2','MY7','ST5','TH4']            
ld_X_aq_list = ['NO2', 'PM10', 'PM2.5']  
ld_y_aq_list = ['PM10', 'PM2.5'] 
ld_X_meo_list = ["temperature","pressure","humidity","direction","speed"]


bj_near_stations = {'aotizhongxin_aq': 'beijing_grid_304',
                     'badaling_aq': 'beijing_grid_224',
                     'beibuxinqu_aq': 'beijing_grid_263',
                     'daxing_aq': 'beijing_grid_301',
                     'dingling_aq': 'beijing_grid_265',
                     'donggaocun_aq': 'beijing_grid_452',
                     'dongsi_aq': 'beijing_grid_303',
                     'dongsihuan_aq': 'beijing_grid_324',
                     'fangshan_aq': 'beijing_grid_238',
                     'fengtaihuayuan_aq': 'beijing_grid_282',
                     'guanyuan_aq': 'beijing_grid_282',
                     'gucheng_aq': 'beijing_grid_261',
                     'huairou_aq': 'beijing_grid_349',
                     'liulihe_aq': 'beijing_grid_216',
                     'mentougou_aq': 'beijing_grid_240',
                     'miyun_aq': 'beijing_grid_392',
                     'miyunshuiku_aq': 'beijing_grid_414',
                     'nansanhuan_aq': 'beijing_grid_303',
                     'nongzhanguan_aq': 'beijing_grid_324',
                     'pingchang_aq': 'beijing_grid_264',
                     'pinggu_aq': 'beijing_grid_452',
                     'qianmen_aq': 'beijing_grid_303',
                     'shunyi_aq': 'beijing_grid_368',
                     'tiantan_aq': 'beijing_grid_303',
                     'tongzhou_aq': 'beijing_grid_366',
                     'wanliu_aq': 'beijing_grid_283',
                     'wanshouxigong_aq': 'beijing_grid_303',
                     'xizhimenbei_aq': 'beijing_grid_283',
                     'yanqin_aq': 'beijing_grid_225',
                     'yizhuang_aq': 'beijing_grid_323',
                     'yongdingmennei_aq': 'beijing_grid_303',
                     'yongledian_aq': 'beijing_grid_385',
                     'yufa_aq': 'beijing_grid_278',
                     'yungang_aq': 'beijing_grid_239',
                     'zhiwuyuan_aq': 'beijing_grid_262'}


ld_near_stations = {'BL0': 'london_grid_409',
                     'BX1': 'london_grid_472',
                     'BX9': 'london_grid_472',
                     'CD1': 'london_grid_388',
                     'CD9': 'london_grid_409',
                     'CR8': 'london_grid_408',
                     'CT2': 'london_grid_409',
                     'CT3': 'london_grid_409',
                     'GB0': 'london_grid_451',
                     'GN0': 'london_grid_451',
                     'GN3': 'london_grid_451',
                     'GR4': 'london_grid_451',
                     'GR9': 'london_grid_430',
                     'HR1': 'london_grid_368',
                     'HV1': 'london_grid_472',
                     'KC1': 'london_grid_388',
                     'KF1': 'london_grid_388',
                     'LH0': 'london_grid_346',
                     'LW2': 'london_grid_430',
                     'MY7': 'london_grid_388',
                     'RB7': 'london_grid_452',
                     'ST5': 'london_grid_408',
                     'TD5': 'london_grid_366',
                     'TH4': 'london_grid_430'}