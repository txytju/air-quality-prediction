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

day = day - 1
# 下载数据

url_bj_aq = 'https://biendata.com/competition/airquality/bj/2018-05-%d-0/2018-05-%d-23/2k0d1d8' %(day, day)
respones= requests.get(url_bj_aq)
name = "./Beijing/aq/bj_aq_2018-05-%d-0_2018-05-%d-23.csv" %(day, day)
with open (name,'w') as f:
	f.write(respones.text)
print("done ", name)


url_bj_meo = 'https://biendata.com/competition/meteorology/bj_grid/2018-05-%d-0/2018-05-%d-23/2k0d1d8' %(day, day)
respones= requests.get(url_bj_meo)
name = "./Beijing/grid_meo/bj_grid_2018-05-%d-0_2018-05-%d-23.csv" %(day, day)
with open (name,'w') as f:
	f.write(respones.text)
print("done ", name)


url_ld_aq = 'https://biendata.com/competition/airquality/ld/2018-05-%d-0/2018-05-%d-23/2k0d1d8' %(day, day)
respones= requests.get(url_ld_aq)
name = "./London/aq/ld_aq_2018-05-%d-0_2018-05-%d-23.csv" %(day, day)
with open (name,'w') as f:
	f.write(respones.text)
print("done ", name)


url_ld_meo = 'https://biendata.com/competition/meteorology/ld_grid/2018-05-%d-0/2018-05-%d-23/2k0d1d8' %(day, day)
respones= requests.get(url_ld_meo)
name = "./London/grid_meo/ld_grid_2018-05-%d-0_2018-05-%d-23.csv" %(day, day)
with open (name,'w') as f:
	f.write(respones.text)
print("done ", name)










