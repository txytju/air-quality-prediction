import requests
import datetime


# 获得当前的 UTC 时间
now = datetime.datetime.utcnow()

# 获得一天前和现在的时间
# year_s = ago.year
# month_s = ago.month
# day_s = ago.day
# hour_s = ago.hour

# year_e = now.year
# month_e = now.month
day = now.day
hour = now.hour

print("day is :", day, "hour is : ", hour)

outfileName = './test/outcid.txt'  
outfile = open(outfileName, 'w')  
outfile.write("This is crontab command test!")  
outfile.close()

# 下载数据

# url_bj_aq = 'https://biendata.com/competition/airquality/bj/2018-05-%d-0/2018-05-%d-23/2k0d1d8' %(day, day)
# respones= requests.get(url_bj_aq)
# name = "./test/Beijing/aq/bj_aq_2018-05-%d-0_2018-05-%d-23.csv" %(day, day)
# with open (name,'w') as f:
# 	f.write(respones.text)
# print("done ", name)


# url_bj_meo = 'https://biendata.com/competition/meteorology/bj_grid/2018-05-%d-0/2018-05-%d-23/2k0d1d8' %(day, day)
# respones= requests.get(url_bj_meo)
# name = "./test/Beijing/grid_meo/bj_grid_2018-05-%d-0_2018-05-%d-23.csv" %(day, day)
# with open (name,'w') as f:
# 	f.write(respones.text)
# print("done ", name)


# url_ld_aq = 'https://biendata.com/competition/airquality/ld/2018-05-%d-0/2018-05-%d-23/2k0d1d8' %(day, day)
# respones= requests.get(url_ld_aq)
# name = "./test/London/aq/ld_aq_2018-05-%d-0_2018-05-%d-23.csv" %(day, day)
# with open (name,'w') as f:
# 	f.write(respones.text)
# print("done ", name)


# url_ld_meo = 'https://biendata.com/competition/meteorology/ld_grid/2018-05-%d-0/2018-05-%d-23/2k0d1d8' %(day, day)
# respones= requests.get(url_ld_meo)
# name = "./test/London/grid_meo/ld_grid_2018-05-%d-0_2018-05-%d-23.csv" %(day, day)
# with open (name,'w') as f:
# 	f.write(respones.text)
# print("done ", name)















# 	url = 'https://biendata.com/competition/meteorology/bj_grid/2018-04-%s-0/2018-04-%s-23/2k0d1d8' %(i_str, i_str)
# 	urls.append(url)

# for url in urls :
# 	print(url)

# j = 0
# for url in urls :
# 	respones= requests.get(url)
# 	name = "new_%s.csv" %(str(j))
# 	with open (name,'w') as f:
# 		f.write(respones.text)
# 	j += 1
# 	print("done!")

# print("done !!")

# Air Quality data
# https://biendata.com/competition/airquality/bj/2018-04-01-0/2018-04-01-23/2k0d1d8

# Observed Meteorology:
# https://biendata.com/competition/meteorology/bj/2018-04-01-0/2018-04-01-23/2k0d1d8

# Meteorology Grid Data:
# https://biendata.com/competition/meteorology/bj_grid/2018-04-01-0/2018-04-01-23/2k0d1d8










