from SearchingAPI import *
from BuildingProcessing import *
import shutil
import os
import pandas as pd
import time
from pyproj import CRS
from pyproj import Transformer

#2020-09-10. 이사님 작업보조로 인한 코드 수정의 건.
def get_address_original_codes(processed_address):
    road_result = road_search(processed_address)
    if len(road_result['results']['juso']) == 0:
        raise Exception("잘못된 주소를 입력하였습니다. 주소값이 0입니다.")

    def get_road_code(index):
        return road_result['results']['juso'][index]

    rode_code = get_road_code(0)
    # mncode : 건물관리번호
    mncode = rode_code['bdMgtSn']

    coord_result = coord_search(rode_code['admCd'], rode_code['rnMgtSn'], rode_code['udrtYn'], rode_code['buldMnnm'], rode_code['buldSlno'])
    print(coord_result)
    if len(coord_result['results']['juso']) == 0:
        raise Exception("잘못된 코드를 입력하였습니다. 결과값이 0입니다.")

    def get_coord_code(index):
        return coord_result['results']['juso'][index]

    coord_code = get_coord_code(0)
    #x좌표, y좌표
    #x_coord_temp = coord_code['entX']
    #y_coord_temp = coord_code['entY']


    crs_5179 = CRS.from_epsg(5179)
    crs_4326 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_5179, crs_4326, always_xy=True)
    x_coord = transformer.transform(coord_code['entX'], coord_code['entY'])[0]
    y_coord = transformer.transform(coord_code['entX'], coord_code['entY'])[1]




    bldgname = coord_code['bdNm']

    return mncode, bldgname, x_coord, y_coord

def make_excel_coord(all_infos, columns):
    index_format = [index+1 for index, all_info in enumerate(all_infos)]
    #columns = ['raw_address', 'processed_address', 'address', 'sggu_code', 'Result', 'Ref']

    # DataFrame 초기화
    values = pd.DataFrame(index=index_format, columns=columns)
    row_numbers = values.shape[0]
    index_row = index_format

    for index in range(row_numbers):
        values.iloc[index, 0] = index_row[index]
        for column_index, column in enumerate(columns):
            values.iloc[index, column_index] = all_infos[index].get(column)

    # saves DataFrame(values) into an Excel file
    values.to_excel('./weather_result.xlsx',
                    sheet_name='Sheet1',
                    columns=columns,
                    header=True,
                    index=index_format,
                    startrow=1,
                    startcol=0,
                    engine=None,
                    merge_cells=True,
                    encoding=None,
                    inf_rep='inf',
                    verbose=True,
                    freeze_panes=None)
'''
admCd = 2914011800
rnMgtSn = 291403160032
udrtYn = 0
buldMnnm = 47
buldSlno = 0
k = coord_search(str(admCd), str(rnMgtSn), str(udrtYn), str(buldMnnm), str(buldSlno))
print(k)
'''

#임시로 만든 것임 (2020-09-10)
#파일 소스 읽는 기능은 자체적으로 넣기.
data = pd.read_csv('./cluster_20210911/8 clusters_good/original_data2.csv', header=0, encoding='ANSI')
filtered_data = data[['bjdCode']]
filtered_lists = filtered_data["bjdCode"].tolist()

all_coord_infos = []

for item in filtered_lists:
    temp_coord_info = {}
    temp_coord_info['bjdcode'] = item
    sggcode = int(str(item)[0:5])
    if sggcode < 20000:
        temp_coord_info['Area'] = '서울경기'
    elif sggcode < 27000:
        temp_coord_info['Area'] = '경남'
    elif sggcode < 28000:
        temp_coord_info['Area'] = '경북'
    elif sggcode < 29000:
        temp_coord_info['Area'] = '서울경기'
    elif sggcode < 30000:
        temp_coord_info['Area'] = '전남'
    elif sggcode < 31000:
        temp_coord_info['Area'] = '충북'
    elif sggcode < 36110:
        temp_coord_info['Area'] = '경남'
    elif sggcode < 41000:
        temp_coord_info['Area'] = '충남'
    elif sggcode < 42000:
        temp_coord_info['Area'] = '서울경기'
    elif sggcode < 42150:
        temp_coord_info['Area'] = '강원영서'
    elif sggcode < 42720:
        temp_coord_info['Area'] = '강원영동'
    elif sggcode < 42820:
        temp_coord_info['Area'] = '강원영서'
    elif sggcode < 43000:
        temp_coord_info['Area'] = '강원영동'
    elif sggcode < 44000:
        temp_coord_info['Area'] = '충북'
    elif sggcode < 45000:
        temp_coord_info['Area'] = '충남'
    elif sggcode < 46000:
        temp_coord_info['Area'] = '전북'
    elif sggcode < 47000:
        temp_coord_info['Area'] = '전남'
    elif sggcode < 48000:
        temp_coord_info['Area'] = '경북'
    elif sggcode < 50000:
        temp_coord_info['Area'] = '경남'
    elif sggcode >= 50000:
        temp_coord_info['Area'] = '제주'

    weather_data = pd.read_csv('./cluster_20210911/8 clusters_good/temperature_annual.csv', header=0, encoding='ANSI')
    weather_arr = weather_data.values
    #print(type(weather_arr))
    #print(weather_arr)
    for region in weather_arr:
        if region[0] == temp_coord_info['Area']:
            temp_coord_info['Temp_avg'] = region[1]
            temp_coord_info['Temp_min'] = region[2]
            temp_coord_info['Temp_max'] = region[3]

    all_coord_infos.append(temp_coord_info)
print(all_coord_infos)
rowlist_for_excel = all_coord_infos[0].keys()
make_excel_coord(all_coord_infos, rowlist_for_excel)
