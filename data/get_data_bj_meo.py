import requests



url = 'https://biendata.com/competition/meteorology/bj_grid/2018-04-30-0/2018-04-30-23/2k0d1d8'
respones= requests.get(url)
name = "./Beijing/grid_meo/bj_grid_2018-04-30-0_2018-04-30-23.csv"
with open (name,'w') as f:
	f.write(respones.text)
print("done ", name)











