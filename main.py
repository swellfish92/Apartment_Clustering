#!/usr/bin/env python
# -*- coding: utf-8 -*-

from SearchingAPI import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#건축물대장 참조를 위해 추가된 것들.
def get_address_codes(processed_address):
    road_result = road_search(processed_address)
    if len(road_result['results']['juso']) == 0:
        raise Exception("잘못된 주소를 입력하였습니다. 주소값이 0입니다.")

    def get_road_code(index):
        return road_result['results']['juso'][index]

    rode_code = get_road_code(0)

    # sggcode : 시군구 코드
    sgg_code = rode_code['admCd'][0:5]

    # bjdcode : 법정도 코드
    bjd_code = str(rode_code['admCd'][5:10])

    # buncode : 번 코드
    bun_code = rode_code['lnbrMnnm'].zfill(4)

    # jicode : 지 코드
    ji_code = rode_code['lnbrSlno'].zfill(4)

    # mountain_code : 대지구분코드. 2020.10.29 추가됨.
    mountain_code = str(rode_code['mtYn'])

    # pnu : 건물일련번호. 2020.12.29 추가됨.
    pnu = sgg_code + bjd_code +str(int(mountain_code) + 1) + bun_code + ji_code

    #도로명주소 텍스트. 2021.01.04 추가됨.
    roadaddr = rode_code['roadAddrPart1']

    print(sgg_code, bjd_code, bun_code, ji_code, mountain_code, pnu, roadaddr)
    return sgg_code, bjd_code, bun_code, ji_code, mountain_code, pnu, roadaddr

def get_land_price(land_price_result):
    #print(land_price_result)
    if land_price_result['response']['totalCount'] == 1:
        return land_price_result['response']['fields']['field'][0]
    elif land_price_result['response']['totalCount'] == 0:
        return "No result"
    else:
        temp_year = 0
        count = 0
        for item in land_price_result['response']['fields']['field']:
            if int(item['stdrYear']) > temp_year:
                temp_year = int(item['stdrYear'])
                max_count = count
            count = count + 1
        return land_price_result['response']['fields']['field'][max_count]

def cluster_graph(var_one, var_two, var_char):
    ax = sns.scatterplot(x=var_one, y=var_two, hue=var_char)
    # plt.title("Iris 데이터 중, 꽃잎의 길이에 대한 Kernel Density Plot")
    plt.show()


'''data = read_text_file("list.txt", 15863)

all_infos = []
for item in data:
    temp_info = {}
    item_split = item.split(',')
    print(item_split)
    try:
        processed_address = get_infos(item_split[0])
        print(processed_address)
    except:
        temp_info['result'] = '주소정제 오류'
        all_infos.append(temp_info)
        continue

    try:
        (sgg_code, bjd_code, bun_code, ji_code, mountain_code, pnu, roadaddr) = get_address_codes(processed_address)
        print(pnu)
    except:
        temp_info['result'] = '행안부 검색오류'
        all_infos.append(temp_info)
        continue

    try:
        landprice_result = land_price_search(pnu)
        final_landprice = get_land_price(landprice_result)
        print(final_landprice)
        temp_info['pblntfPclnd'] = final_landprice['pblntfPclnd']

    except:
        temp_info['result'] = '공시지가 검색오류'
        all_infos.append(temp_info)
        continue

    temp_info["Result"] = "SUCCESS"
    all_infos.append(temp_info)'''

#make_excel_school(var, 'result', ['pblntfPclnd', 'Result'])
import numpy as np


def bounding_box(x1, y1, x2, y2, up_thickness, down_thickness):
    if x1 > x2:
        return bounding_box(x2, y2, x1, y1, up_thickness, down_thickness)
    theta = np.arctan((y2-y1)/(x2-x1))
    print(theta)
    top_x_diff = np.sin(theta + np.pi/2) * up_thickness
    top_y_diff = np.cos(theta + np.pi/2) * up_thickness
    bot_x_diff = np.sin(theta - np.pi/2) * down_thickness
    bot_y_diff = np.cos(theta - np.pi/2) * down_thickness
    result_point_1 = (x1 + top_x_diff, y1 + top_y_diff)
    result_point_2 = (x1 + bot_x_diff, y1 + bot_y_diff)
    result_point_4 = (x2 + top_x_diff, y2 + top_y_diff)
    result_point_3 = (x2 + bot_x_diff, y2 + bot_y_diff)
    return (result_point_1, result_point_2, result_point_3, result_point_4)

#print(bounding_box(0, 0, -50, -50, 2, 0))


import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

#직교좌표를 극좌표로 변형 (0~2pi 범위)
def cart_to_polar(x1, y1, x2, y2):
    if x2-x1 > 0 and y2-y1 >= 0:
        return np.arctan((y2-y1)/(x2-x1))
    elif x2-x1 > 0 and y2-y1 < 0:
        return np.arctan((y2-y1)/(x2-x1)) + 2*np.pi
    elif x2-x1 < 0:
        return np.arctan((y2-y1)/(x2-x1)) + np.pi
    elif x2-x1 == 0 and y2-y1 > 0:
        return np.pi/2
    elif x2-x1 == 0 and y2-y1 < 0:
        return 3*np.pi/2
    else:
        print("ERROR (극좌표계 r값이 0입니다.)")
        return np.nan

def bounding_box2(x1, y1, x2, y2, center_x, center_y, thickness):
    #중심점 각도 구하기
    center_angle = cart_to_polar(x1, y1, center_x, center_y)

    #1번 점과 2번 점의 각도 구하기
    line_angle = cart_to_polar(x1, y1, x2, y2)

    #line_angle이 pi를 넘어갈 경우는 pi를 빼 준다 (호도법 기준 직선각 210도를 30도로 바꿔 주는 부분)
    if line_angle >= np.pi:
        line_angle = line_angle - np.pi
    #center point까지의 각이 line의 각보다 크고, line의 각 + 180도보다 적은 경우 -> 마이너스 1/2파이만큼 각을 아래로 틀어서 offset. 반대는 위로 틀어서 offset.
    if center_angle > line_angle and center_angle < line_angle + np.pi:
        theta = line_angle - np.pi/2
    elif center_angle == line_angle or center_angle == line_angle + np.pi:
        print("ERROR (중심점이 폴리곤의 연장선상에 위치합니다.)")
        return np.nan
    else:
        theta = line_angle + np.pi / 2

    result_point_1 = [x1, y1]
    result_point_2 = [x1 + thickness * np.cos(theta), y1 + thickness * np.sin(theta)]
    result_point_3 = [x2 + thickness * np.cos(theta), y2 + thickness * np.sin(theta)]
    result_point_4 = [x2, y2]
    return [result_point_1, result_point_2, result_point_3, result_point_4]



first_point = [0, 0]
second_point = [50, -30]
center_point = [40, -50]

print(bounding_box2(first_point[0], first_point[1], second_point[0], second_point[1], center_point[0], center_point[1], 10))
points = bounding_box2(first_point[0], first_point[1], second_point[0], second_point[1], center_point[0], center_point[1], 10)
shp = patches.Polygon(points)
plt.gca().add_patch(shp)
plt.scatter(center_point[0], center_point[1], color='r')
plt.plot([first_point[0], second_point[0]], [first_point[1], second_point[1]], color='red')
plt.axis('scaled')
plt.show()