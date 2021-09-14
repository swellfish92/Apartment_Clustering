from SearchingAPI import *


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

    #sgg_code = str(rode_code['bdMgtSn'][0:5])
    #bjd_code = str(rode_code['bdMgtSn'][5:10])
    #bun_code = str(rode_code['bdMgtSn'][11:15])
    #ji_code = str(rode_code['bdMgtSn'][15:19])

    print(sgg_code, bjd_code, bun_code, ji_code, mountain_code, pnu)
    return sgg_code, bjd_code, bun_code, ji_code, mountain_code, pnu

#건물이름 비교를 위해 임시추가. 2020.11.02
def get_address_codes_bldgname_included(processed_address):
    road_result = road_search(processed_address)
    if len(road_result['results']['juso']) == 0:
        raise Exception("잘못된 주소를 입력하였습니다. 주소값이 0입니다.")

    def get_road_code(index):
        return road_result['results']['juso'][index]

    rode_code = get_road_code(0)

    # sggcode : 시군구 코드
    #sgg_code = rode_code['admCd'][0:5]

    # bjdcode : 법정도 코드
    #bjd_code = str(rode_code['admCd'][5:10])

    # buncode : 번 코드
    #bun_code = rode_code['lnbrMnnm'].zfill(4)

    # jicode : 지 코드
    #ji_code = rode_code['lnbrSlno'].zfill(4)

    # mountain_code : 대지구분코드. 2020.10.29 추가됨.
    mountain_code = rode_code['mtYn']

    sgg_code = str(rode_code['bdMgtSn'][0:5])
    bjd_code = str(rode_code['bdMgtSn'][5:10])
    bun_code = str(rode_code['bdMgtSn'][11:15])
    ji_code = str(rode_code['bdMgtSn'][15:19])
    #mountain_code = str(rode_code['bdMgtSn'][10:11])
    building_name = str(rode_code['bdNm'])
    print(sgg_code, bjd_code, bun_code, ji_code, mountain_code, building_name)
    return sgg_code, bjd_code, bun_code, ji_code, mountain_code, building_name


# mountain_code : 대지구분코드. 2020.10.29 추가됨.
def get_building_result_object(sgg_code, bjd_code, bun_code, ji_code, mountain_code):
    def find_bun_ji():
        subjibun_result_total_count = subjibun_search(sgg_code, bjd_code, 1)['response']['body']['totalCount']
        page_numbers = int(subjibun_result_total_count) // 999 + 1
        for page_number in range(1, page_numbers + 1):
            items = subjibun_search(sgg_code, bjd_code, page_number)['response']['body']['items']['item']
            for item in items:
                if item.get('atchBun') == bun_code and item.get('atchJi') == ji_code:
                    return item.get('bun'), item.get('ji')

    building_result = building_search(sgg_code, bjd_code, bun_code, ji_code, mountain_code)
    if int(building_result['response']['body']['totalCount']) == 0:
        # result = ('1164', '0008') 항상 튜플값으로 반환(지, 번)
        result = find_bun_ji()
        if result is None:
            raise Exception("서브지번으로 검색했지만 일치하는 번지가 없습니다.")
        building_result = building_search(sgg_code, bjd_code, result[0], result[1])

    return building_result

#부속건축물 여부 찾는 함수. 동일하게 시군구-법정동 내에서 찾은 뒤 주부속구분코드(mainAtchGbCd)가 0(주건축물)인 아이템을 배열로 돌려줌
def find_subbuilding(sgg_code, bjd_code):
    subbuilding_result_total_count = subbuilding_search(sgg_code, bjd_code, 1)['response']['body']['totalCount']
    page_numbers = int(subbuilding_result_total_count) // 999 + 1
    subbuilding_list = list()
    for page_number in range(1, page_numbers + 1):
        items = subbuilding_search(sgg_code, bjd_code, page_number)['response']['body']['items']['item']
        for item in items:
            if item['mainAtchGbCd'] == '0':
                subbuilding_list.append(item['mgmBldrgstPk'])
    print(subbuilding_list)
    return subbuilding_list


def get_final_infos(building_result):
    response = {}
    building_result_total_count = int(building_result['response']['body']['totalCount'])
    key_list = ['sigunguCd', 'bjdongCd', 'bun', 'ji', 'mgmBldrgstPk', 'mgmUpBldrgstPk']


    def save_response(item):
        for key in key_list:
            response[key] = item.get(key)
        return response



    if building_result_total_count == 1:
        return save_response(building_result['response']['body']['items']['item'])
    elif building_result_total_count > 1:
        temp_response_objects = []
        for item in building_result['response']['body']['items']['item']:
            if (item['regstrGbCd'] == "1" and item['regstrKindCd'] == "2") or (item['regstrGbCd'] == "2" and item['regstrKindCd'] == "3"):
                temp_response_objects.append(item)
        if len(temp_response_objects) == 1:
            return save_response(temp_response_objects[0])
        else:
            temp_response_objects = []
            for item in building_result['response']['body']['items']['item']:
                if item['regstrKindCd'] == "1":
                    return save_response(item)


            #집합건축물의 총괄표제부는 항상 있다고 가정함. 없는 케이스가 아직 안 나왔으니...
            #주건축물 목록을 받아와 저장함.
            mainbuilding_list = find_subbuilding(item['sigunguCd'], item['bjdongCd'])
            for item in building_result['response']['body']['items']['item']:

            #API호출이 많아질 것 같기도 한데... 몇 개 없겠지?
            #주건축물 목록에 있는지 확인
                if item['mgmBldrgstPk'] in mainbuilding_list:
                    temp_response_objects.append(item)

            if len(temp_response_objects) == 1:
                print("주건축물 선정 리스트")
                print(temp_response_objects)
                return save_response(temp_response_objects[0])

            else:
                print("주건축물 선정 리스트")
                print(temp_response_objects)
                return "Multiple Case"




    else:
        return "Error"

