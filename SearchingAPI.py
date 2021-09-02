#from googleapiclient.discovery import build
#from google_auth_oauthlib.flow import InstalledAppFlow
#import google.oauth2.credentials
import requests
import json
import xmltodict
import os
import copy
import pandas as pd
import numpy as np
from RoadProcessing import *

CLIENT_SECRETS_FILE = "credentials.json"
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
API_SERVICE_NAME = 'sheets'
API_VERSION = 'v4'


def get_sheets():
    credential_path = os.path.join("./", 'sheets.googleapis.com-python-quickstart.json')
    if os.path.exists(credential_path):
        with open(credential_path, 'r') as f:
            credential_params = json.load(f)
        credentials = google.oauth2.credentials.Credentials(
            token=credential_params["token"],
            refresh_token=credential_params["_refresh_token"],
            token_uri=credential_params["_token_uri"],
            client_id=credential_params["_client_id"],
            client_secret=credential_params["_client_secret"]
        )
    else:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
        credentials = flow.run_console()
        with open(credential_path, 'w') as f:
            p = copy.deepcopy(vars(credentials))
            del p["expiry"]
            json.dump(p, f, indent=4)
    return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)


def get_spreadsheet_value(sheet, spreadsheet_id, range_str):
    return sheet.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=range_str).execute().get('values', [])


def update_spreadsheet(sheet, spreadsheet_id, range_str, values):
    body = {'values': values}
    sheet.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id, range=range_str,
        valueInputOption='USER_ENTERED', body=body).execute()

#--------------------------------------------------------------------------------------------------------
#OpenAPI 콜에 관련된 것들. 인증키 만료는 이쪽을 참조.


#http://apis.data.go.kr/1611000/nsdi/IndvdLandPriceService/attr/getIndvdLandPriceAttr?ServiceKey=인증키&pnu=1111017700102110000&stdrYear=2015&format=xml&numOfRows=10&pageNo=1
def land_price_search(pnu):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/1611000/nsdi/IndvdLandPriceService/attr/getIndvdLandPriceAttr?ServiceKey="+ str(key) + "&pnu=" + str(pnu) + "&format=xml&numOfRows=999&pageNo=1"
    print(url)
    result = requests.get(url)
    jsonresult = json.loads(json.dumps(xmltodict.parse(result.text)))
    return jsonresult

#2020-09-10. 좌표서치
def coord_search(admCd, rnMgtSn, udrtYn, buldMnnm, buldSlno):
    key = "devU01TX0FVVEgyMDIwMDkxMDExNTMzNjExMDE2NjI="
    url = "http://www.juso.go.kr/addrlink/addrCoordApi.do?&resultType=json&admCd=" + admCd + "&rnMgtSn=" + rnMgtSn + "&udrtYn=" + udrtYn + "&buldMnnm=" + buldMnnm + "&buldSlno=" + buldSlno + "&confmKey=" + key
    result = requests.get(url)
    return result.json()


def road_search(srctext):
    key = "devU01TX0FVVEgyMDIxMDcwNzE2Mjg1MDExMTM3ODQ="
    url = "http://www.juso.go.kr/addrlink/addrLinkApi.do?&resultType=json&currentPage=1&countPerPage=100&confmKey=" + key + "&keyword=" + srctext
    print(url)
    result = requests.get(url)
    return result.json()

#2020.10.29 - 산여부(대지구분코드-platGbCd, 내부변수명 mountain)을 추가하였음.
def building_search(sgg, bjd, bun=None, ji=None, mountain=None):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    if bun is not None and ji is not None and mountain is not None:
        url = "http://apis.data.go.kr/1611000/BldRgstService/getBrBasisOulnInfo?sigunguCd=" + sgg + "&bjdongCd=" + bjd + "&bun=" + bun + "&ji=" + ji + "&numOfRows=999" + "&platGbCd=" + mountain + "&ServiceKey=" + key
    else:
        url = "http://apis.data.go.kr/1611000/BldRgstService/getBrBasisOulnInfo?sigunguCd=" + sgg + "&bjdongCd=" + bjd + "&numOfRows=999" + "&ServiceKey=" + key
    print(url)
    result = requests.get(url)
    jsonresult = json.loads(json.dumps(xmltodict.parse(result.text)))
    return jsonresult

#2번 오퍼레이션에 대한 API호출함수
def building_search_op2(sgg, bjd, bun, ji):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/1611000/BldRgstService/getBrRecapTitleInfo?sigunguCd=" + sgg + "&bjdongCd=" + bjd + "&bun=" + bun + "&ji=" + ji + "&numOfRows=999" + "&ServiceKey=" + key
    print(url)
    result = requests.get(url)
    jsonresult = json.loads(json.dumps(xmltodict.parse(result.text)))
    return jsonresult

#3번 오퍼레이션에 대한 API호출함수
def building_search_op3(sgg, bjd, bun, ji):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/1611000/BldRgstService/getBrTitleInfo?sigunguCd=" + sgg + "&bjdongCd=" + bjd + "&bun=" + bun + "&ji=" + ji + "&numOfRows=999" + "&ServiceKey=" + key
    print(url)
    result = requests.get(url)
    jsonresult = json.loads(json.dumps(xmltodict.parse(result.text)))
    return jsonresult

#산여부 제외한 호출함수
def building_search_gen(sgg, bjd, bun, ji):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/1611000/BldRgstService/getBrBasisOulnInfo?sigunguCd=" + sgg + "&bjdongCd=" + bjd + "&bun=" + bun + "&ji=" + ji + "&numOfRows=999" + "&ServiceKey=" + key
    print(url)
    result = requests.get(url)
    jsonresult = json.loads(json.dumps(xmltodict.parse(result.text)))
    return jsonresult


def subjibun_search(sgg, bjd, page):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/1611000/BldRgstService/getBrAtchJibunInfo?sigunguCd=" + sgg + "&bjdongCd=" + bjd + "&ServiceKey=" + key + "&numOfRows=999&pageNo=" + str(page)
    result = requests.get(url)
    jsonresult = json.loads(json.dumps(xmltodict.parse(result.text)))
    return jsonresult

def subbuilding_search(sgg, bjd, page):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/1611000/BldRgstService/getBrTitleInfo?sigunguCd=" + sgg + "&bjdongCd=" + bjd + "&ServiceKey=" + key + "&numOfRows=999&pageNo=" + str(page)
    result = requests.get(url)
    jsonresult = json.loads(json.dumps(xmltodict.parse(result.text)))
    return jsonresult

#학교 프로젝트로 인해 추가됨(2020.12.29). API의 증가로 주석을 달기로 했음...앞에 것들은 귀찮으니 패스.
#건물에너지 API (에너지공단-Data.go.kr)
def energy_search_elec(sgg, bjd, bun, ji, YM):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/1611000/BldEngyService/getBeElctyUsgInfo?sigunguCd=" + sgg + "&bjdongCd=" + bjd + "&bun=" + bun + "&ji=" + ji + "&useYm=" + YM + "&ServiceKey=" + key
    print(url)
    result = requests.get(url)
    jsonresult = json.loads(json.dumps(xmltodict.parse(result.text)))
    return jsonresult

#건물에너지 API (에너지공단-Data.go.kr): 활용을 위해 gas와 elec으로 별도로 분류함.
def energy_search_gas(sgg, bjd, bun, ji, YM):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/1611000/BldEngyService/getBeGasUsgInfo?sigunguCd=" + sgg + "&bjdongCd=" + bjd + "&bun=" + bun + "&ji=" + ji + "&useYm=" + YM + "&ServiceKey=" + key
    print(url)
    result = requests.get(url)
    jsonresult = json.loads(json.dumps(xmltodict.parse(result.text)))
    return jsonresult

#건축물연령정보서비스 API(국토교통부-Data.go.kr) 현재는 선언만 해두고 보류하도록 (자체적으로 부속지번 검색 알고리즘이 필요한데, 기준이 없음!)
def building_age_search(pnu):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/1611000/nsdi/BuildingAgeService/attr/getBuildingAge?ServiceKey=" + key + "&pnu=" + pnu + "&format=xml&numOfRows=10&pageNo=1"
    result = requests.get(url)
    jsonresult = json.loads(json.dumps(xmltodict.parse(result.text)))
    return jsonresult

def school_search_gen(apicode, year, schoolcode):
    key = "33caef16dd334bc3a086de5f07af5aae"
    schoolcode = str(schoolcode)
    if len(str(schoolcode)) == 1:
        schoolcode = "0" + schoolcode
    url = "http://www.schoolinfo.go.kr/openApi.do?" + "&apiKey=" + key + "&apiType=" + str(apicode) + "&pbanYr=" + str(year) + "&schulKndCode=" + str(schoolcode)
    print("url is : " + url)
    result = requests.get(url, verify=False)
    return result.json()

#아파트 프로젝트로 인해 추가됨(2021.03.25)
#아파트 단지코드 검색 API (K-APT:Data.go.kr)
def aptcode_search(type, code):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    if type == 1:
        url = "http://apis.data.go.kr/1613000/AptListService1/getLegaldongAptList?bjdCode=" + str(code) + "&numOfRows=999&ServiceKey=" + key
    elif type == 2:
        url = "http://apis.data.go.kr/1613000/AptListService1/getSigunguAptList?sigunguCode=" + str(code) + "&numOfRows=999&ServiceKey=" + key
    elif type == 3:
        url = "http://apis.data.go.kr/1613000/AptListService1/getSidoAptList?sidoCode=" + str(code) + "&numOfRows=999&ServiceKey=" + key
    elif type == 4:
        #전국은 페이지 기능이 추가되어야 하는 것으로 보임. 일단은 code를 페이지로 적자.
        url = "http://apis.data.go.kr/1613000/AptListService1/getTotalAptList?ServiceKey=" + key + "&numOfRows=999&pageNo=" + str(code)
    elif type == 5:
        url = "http://apis.data.go.kr/1613000/AptListService1/getRoadnameAptList?roadCode=" + str(code) + "&numOfRows=999&ServiceKey=" + key
    print(url)
    result = requests.get(url)
    jsonresult = json.loads(json.dumps(xmltodict.parse(result.text)))
    return jsonresult

#아파트 기본정보 검색 API (K-APT:Data.go.kr)
def aptdata_search(code):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/1611000/AptBasisInfoService/getAphusBassInfo?kaptCode=" + str(code) + "&ServiceKey=" + key
    print(url)
    result = requests.get(url)
    jsonresult = json.loads(json.dumps(xmltodict.parse(result.text)))
    return jsonresult

#아파트 에너지정보 검색 API (K-APT:Data.go.kr)
def aptenergy_search(code, YM):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/1611000/ApHusEnergyUseInfoOfferService/getHsmpApHusUsgQtyInfoSearch?kaptCode=" + str(code) + "&reqDate=" + str(YM) + "&ServiceKey=" + key
    print(url)
    result = requests.get(url)
    jsonresult = json.loads(json.dumps(xmltodict.parse(result.text)))
    return jsonresult

#--------------------------------------------------------------------------------------------------------
#excel 파일로 출력하는 부분에 대한 함수들

def make_excel(all_infos):
    index_format = [index+1 for index, all_info in enumerate(all_infos)]
    columns = ['raw_address', 'processed_address', 'sgg_code', 'bjd_code', 'bun_code', 'ji_code',
               'sigunguCd', 'bjdongCd', 'bun', 'ji', 'mgmBldrgstPk', 'mgmUpBldrgstPk', 'Result']

    # DataFrame 초기화
    values = pd.DataFrame(index=index_format, columns=columns)
    row_numbers = values.shape[0]
    index_row = index_format

    for index in range(row_numbers):
        values.iloc[index, 0] = index_row[index]
        for column_index, column in enumerate(columns):
            values.iloc[index, column_index] = all_infos[index].get(column)

    # saves DataFrame(values) into an Excel file
    values.to_excel('./result_춘계학술대회_원본.xlsx',
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

def make_excel_with_mt(all_infos):
    index_format = [index+1 for index, all_info in enumerate(all_infos)]
    columns = ['raw_address', 'processed_address', 'sgg_code', 'bjd_code', 'bun_code', 'ji_code', 'mt_code','building_name',
               'sigunguCd', 'bjdongCd', 'bun', 'ji', 'mgmBldrgstPk', 'mgmUpBldrgstPk', 'Result']

    # DataFrame 초기화
    values = pd.DataFrame(index=index_format, columns=columns)
    row_numbers = values.shape[0]
    index_row = index_format

    for index in range(row_numbers):
        values.iloc[index, 0] = index_row[index]
        for column_index, column in enumerate(columns):
            values.iloc[index, column_index] = all_infos[index].get(column)

    # saves DataFrame(values) into an Excel file
    values.to_excel('./result_mountain_without_bdmgtsn_buildingname_included.xlsx',
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

def make_excel_school(all_infos, filename, columns):
    index_format = [index + 1 for index, all_info in enumerate(all_infos)]
    #columns = ['raw_address', 'processed_address', 'sgg_code', 'bjd_code', 'bun_code', 'ji_code', 'mt_code',
    #           'sigunguCd', 'bjdongCd', 'bun', 'ji', 'energy_consumption_elec', 'energy_consumption_gas', 'platArea', 'archArea', 'totArea', 'BldgAge', 'Result']

    # DataFrame 초기화
    values = pd.DataFrame(index=index_format, columns=columns)
    row_numbers = values.shape[0]
    index_row = index_format

    for index in range(row_numbers):
        values.iloc[index, 0] = index_row[index]
        for column_index, column in enumerate(columns):
            values.iloc[index, column_index] = all_infos[index].get(column)

    # saves DataFrame(values) into an Excel file
    values.to_excel('./' + filename + '.xlsx',
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

#--------------------------------------------------------------------------------------------------------
#여기 아래부터는 심평원 API와 관련된 내용임!
#이 부분은 위에 혼합시키지 않았으나, 다음번 학교 건은 혼합하였음. 때를 봐서 이것도 날짜별로 처리할 것 (2020.12.29)
def get_hospital_number(sgg):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/B551182/hospInfoService/getHospBasisList?pageNo=1&numOfRows=10&_type=json&sgguCd=" + str(sgg) + "&ServiceKey=" + key
    result = requests.get(url)
    jsonresult = json.loads(result.text)
    #print(jsonresult['response']['body']['totalCount'])
    return jsonresult['response']['body']['totalCount']

def get_sggu(sgg):
    #시군구코드변환의 기준이 되는 데이터를 불러와 리스트로 저장함 (sgglist-국토부, sggulist-심평원)
    data = pd.read_csv('hospitalcode.csv', header=0, encoding='utf-8', dtype={'국토부코드명칭':object})

    codelist = data[['코드','명칭','국토부코드명칭']]
    sgglist = codelist['국토부코드명칭'].tolist()
    sggulist = codelist['코드'].tolist()

    #별도 로직에 해당하는 것들 목록. 아래의 시 직할코드가 나오면 분류가 잘못된 것임!!!
    #성남시, 수원시, 안양, 안산시, 고양시, 용인시, 청주시, 천안시, 전주시, 포항시, 창원시
    #성남시(31020), 성남시 수정구(31021), 성남시 중원구(31022), 성남시 분당구(31023)
    #수원시(31010), 수원시 장안구(31011), 수원시 권선구(31012), 수원시 팔달구(31013), 수원시 영통구(31014)
    #안양시(31040), 안양시 만안구(31041), 안양시 동안구(31042)
    #안산시(31090), 안산시 상록구(31091), 안산시 단원구(31092)
    #고양시(31100), 고양시 덕양구(31101), 고양시 일산동구(31103), 고양시 일산서구(31104)
    #용인시(31190), 용인시 처인구(31191), 용인시 기흥구(31192), 용인시 수지구(31193)
    #청주시(33040), 청주시 상당구(33041), 청주시 서원구(33042), 청주시 흥덕구(33043), 청주시 청원구(33044)
    #천안시(34010), 천안시 동남구(34011), 천안시 서북구(34012)
    #전주시(35010), 전주시 완산구(35011), 전주시 덕진구(35012)
    #포항시(37010), 포항시 남구(37011), 포항시 북구(37012)
    #창원시(38110), 창원시 의창구(38111), 창원시 성산구(38112), 창원시 마산합포구(38113), 창원시 마산회원구(38114), 창원시 진해구(38115)
    exceptional_sggcode = ['31020', '31010', '31040', '31090', '31100', '33040', '34010', '35010', '37010', '38110']

    if str(sgg) in exceptional_sggcode:
        return 'Exceptional_sggcode'
    if sgg in sgglist:
        return sggulist[sgglist.index(sgg)]
    else:
        return 'Impaired_sggcode'

def hospitalcode_search(sggu, name):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/B551182/hospInfoService/getHospBasisList?pageNo=1&numOfRows=10&_type=json&sgguCd=" + str(sggu) + "&yadmNm="+ name + "&ServiceKey=" + key
    print(url)
    result = requests.get(url)
    jsonresult = json.loads(result.text)
    return jsonresult['response']

#이름 없이 전체 데이터를 가져오는 것임. 싹 긁어오는 것
def Addr_Hospitalcode_search_forlength(sggu):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/B551182/hospInfoService/getHospBasisList?pageNo=1&numOfRows=10&_type=json&sgguCd=" + str(sggu) + "&ServiceKey=" + key
    result = requests.get(url)
    jsonresult = json.loads(result.text)
    return jsonresult['response']

def Addr_Hospitalcode_search(sggu, pageno):
    key = "owsao31Eak4pE2i8SQkuu6bVXieWxILomFCxifqPQAW4wgbkVJ%2F9X0hIzljulAYMV6KcUZPLla2xAUusk4B1wg%3D%3D"
    url = "http://apis.data.go.kr/B551182/hospInfoService/getHospBasisList?pageNo=" + str(pageno) + "&numOfRows=999&_type=json&sgguCd=" + str(sggu) + "&ServiceKey=" + key
    result = requests.get(url)
    jsonresult = json.loads(result.text)
    return jsonresult['response']

def hospitalcode_search_byjuso(sggu, address):
    print(Addr_Hospitalcode_search_forlength(sggu))
    hospital_total_count = Addr_Hospitalcode_search_forlength(sggu)['body']['totalCount']
    page_numbers = int(hospital_total_count) // 999 + 1
    answer_list = []
    for page_number in range(1, page_numbers + 1):
        items = Addr_Hospitalcode_search(sggu, page_number)['body']['items']['item']
        for item in items:
            loaded_address = get_infos(item.get('addr'))
            address = get_infos(address)
            print(loaded_address + " : " + address)
            #print(address + " :: " + loaded_address)
            if loaded_address == address:
                print(item)
                answer_list.append(item)    #원래 ykiho임! 주소 뽑느라 잠시 바꿈(2021.01.27)

    if len(answer_list) == 0:
        return "Not Found"
    elif len(answer_list) == 1:
        return answer_list[0]
    else:
        return 'Multiple Case with : ' + str(answer_list)

    return 'Exceptional Error'

def has_duplicates(seq):
    return len(seq) != len(set(seq))

def get_address_text(processed_address):
    road_result = road_search(processed_address)
    if len(road_result['results']['juso']) == 0:
        raise Exception("잘못된 주소를 입력하였습니다. 주소값이 0입니다.")
    return road_result['results']['juso'][0]['roadAddr']


def make_excel_hp(all_infos):
    index_format = [index+1 for index, all_info in enumerate(all_infos)]
    columns = ['raw_address', 'processed_address', 'address', 'sggu_code', 'estbDd', 'clCdNm', 'drTotCnt', 'intnCnt', 'resdntCnt', 'ykiho', 'Result', 'Ref']

    # DataFrame 초기화
    values = pd.DataFrame(index=index_format, columns=columns)
    row_numbers = values.shape[0]
    index_row = index_format

    for index in range(row_numbers):
        values.iloc[index, 0] = index_row[index]
        for column_index, column in enumerate(columns):
            values.iloc[index, column_index] = all_infos[index].get(column)

    # saves DataFrame(values) into an Excel file
    values.to_excel('./hospital_Addr_result_20210218.xlsx',
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




