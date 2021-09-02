import re


# 파일 읽어오기 함수
def read_text_file(file_path, line_number):
    response = []
    input_file = open(file_path, 'r')
    for lines in range(int(line_number)):
        line = input_file.readline().rstrip("\n")
        response.append(line.split("\n")[0])
    return response

def write_text_file(file_path, list):
    with open(file_path, "w") as file:
        file.writelines(list)


def sido_abbr_substitute(text):
    substitute_dict = [{"abbr":"서울시", "org":"서울특별시"}, {"abbr":"서울", "org":"서울특별시"},
                       {"abbr": "인천시", "org": "인천광역시"}, {"abbr": "인천", "org": "인천광역시"},
                       {"abbr": "부산시", "org": "부산광역시"}, {"abbr": "부산", "org": "부산광역시"},
                       {"abbr": "대구시", "org": "대구광역시"}, {"abbr": "대구", "org": "대구광역시"},
                       {"abbr": "광주시", "org": "광주광역시"}, {"abbr": "광주", "org": "광주광역시"},
                       {"abbr": "대전시", "org": "대전광역시"}, {"abbr": "대전", "org": "대전광역시"},
                       {"abbr": "경기", "org": "경기도"}, {"abbr": "강원", "org": "강원도"},
                       {"abbr": "충북", "org": "충청북도"}, {"abbr": "충남", "org": "충청남도"},
                       {"abbr": "전북", "org": "전라북도"}, {"abbr": "전남", "org": "전라남도"},
                       {"abbr": "경북", "org": "경상북도"}, {"abbr": "경남", "org": "경상남도"},
                       {"abbr": "세종시", "org": "세종특별자치시"}, {"abbr": "세종", "org": "세종특별자치시"},
                       {"abbr": "제주도", "org": "제주특별자치도"}, {"abbr": "제주", "org": "제주특별자치도"}]
    for item in substitute_dict:
        if text == item['abbr']:
            text = text.replace(item["abbr"],item["org"])
    return text

# 전처리 함수
def get_infos(one_line):

    try:
        # one_line : 전처리 전의 도로명주소(ex : 서울특별시 동작구 보라매로5길 20)
        one_line = one_line.replace("  ", " ")
        #.을 실수로 적은 경우 (왜..?) - 2021.01.07 추가됨 /특문은 스페이싱의 목적으로 쓰는 경우가 있으니 스페이스로 치환해 두도록 한다.
        #one_line = one_line.replace(".", " ")

        #확장적 적용: '-'을 제한 특문 전부 제거 (2021.01.15)
        one_line = re.sub('[=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\[\]\<\>`\'…》]', ' ', one_line)

        # 주소개선 전처리 - 2020.09.07 추가됨
        #공백으로 시작하는 경우 대응

        # ~~~ 1길 처럼 (숫자)길 앞의 공백 에러를 제거
        if bool(re.search("[ ]\d+[길]", one_line)):
            substitute_string_list = re.findall("[ ]\d+[길]", one_line)
            substitute_string = substitute_string_list[0].replace(" ", "")
            one_line = one_line.replace(re.search("[ ]\d+[길]", one_line).group(), substitute_string)
            print('공백 처리 : ' + substitute_string_list[0] + ">>" + substitute_string)

        if bool(re.search("[ ]\d+[번][길]", one_line)):
            substitute_string_list = re.findall("[ ]\d+[번][길]", one_line)
            substitute_string = substitute_string_list[0].replace(" ", "")
            one_line = one_line.replace(re.search("[ ]\d+[번][길]", one_line).group(), substitute_string)
            print('공백 처리 : ' + substitute_string_list[0] + ">>" + substitute_string)

        if bool(re.search("[ ]+[-]", one_line)):
            substitute_string_list = re.findall("[ ]+[-]", one_line)
            one_line = one_line.replace(re.search("[ ]+[-]", one_line).group(), "-")
            print('-(대시) 앞 공백 처리됨')

        if bool(re.search("[-][ ]+", one_line)):
            substitute_string_list = re.findall("[-][ ]+", one_line)
            one_line = one_line.replace(re.search("[-][ ]+", one_line).group(), "-")
            print('-(대시) 뒤 공백 처리됨')
        #강변안길 100-0 과 같은 에러케이스 처리
        if bool(re.search("[-][0]+", one_line)):
            substitute_string_list = re.findall("[-][0]+", one_line)
            one_line = one_line.replace(re.search("[-][0]+", one_line).group(), "-")
            print('-0 을 -로 변환함(주소오기 처리)')

        #one_line의 시/도 기입오기 수정. 이중배열로 하면 되긴 할 텐데, 귀찮으니까 HardCode로 조짐
        # 추가사항->구/군을 남기니까 이걸 굳이 할 필요가 없음...제주도만 해 주자.
        one_line = one_line.replace("제주도", "제주특별자치도")
        #one_line = one_line.replace("세종시", "세종특별자치시")

        # one_line에 포함되어 있는 괄호및 괄호내부의 단어가 있다면 전부 삭제(ex : 강원도 강릉시 경강로 2007(남문동 164-1) ---> 강원도 강릉시 경강로 2007)
        if bool(re.search(r"\([\d\s\w,]+\)", one_line)):
            one_line = one_line.replace(re.search(r"\([\d\s\w,]+\)", one_line).group(), "")

        # unit_list : 대한민국 17개 특별시, 광역시, 도, 자치도, 자치시
        unit_list = {"서울특별시",
                     "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시",
                     "경기도", "강원도", "충청북도", "충청남도", "전라북도", "전라남도", "경상북도", "경상남도",
                     "제주특별자치도",
                     "세종특별자치시"}

        small_unit_list = ['종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구', '도봉구', '노원구', '은평구', '서대문구',
                           '마포구', '양천구', '강서구', '구로구', '금천구', '영등포구', '동작구', '관악구', '서초구', '강남구', '송파구', '강동구', '서구', '동구',
                           '영도구', '부산진구', '동래구', '남구', '북구', '해운대구', '사하구', '금정구', '연제구', '수영구', '사상구', '기장군', '수성구', '달서구',
                           '달성군', '중구', '동구', '연수구', '남동구', '부평구', '계양구', '서구', '미추홀구', '강화군', '옹진군', '남구', '북구', '광산구', '유성구',
                           '대덕구', '울주군', '수원시', '장안구', '권선구', '팔달구', '영통구', '성남시', '수정구', '중원구', '분당구', '의정부시', '안양시',
                           '만안구', '동안구', '부천시', '광명시', '평택시', '동두천시', '안산시', '상록구', '단원구', '고양시', '덕양구', '일산동구', '일산서구',
                           '과천시', '구리시', '남양주시', '오산시', '시흥시', '군포시', '의왕시', '하남시', '용인시', '처인구', '기흥구', '수지구', '파주시', '이천시',
                           '안성시', '김포시', '화성시', '광주시', '양주시', '포천시', '여주시', '연천군', '가평군', '양평군', '춘천시', '원주시', '강릉시', '동해시',
                           '태백시', '속초시', '삼척시', '홍천군', '횡성군', '영월군', '평창군', '정선군', '철원군', '화천군', '양구군', '인제군', '고성군', '양양군',
                           '충주시', '제천시', '청주시', '상당구', '서원구', '흥덕구', '청원구', '보은군', '옥천군', '영동군', '진천군', '괴산군', '음성군', '단양군',
                           '증평군', '천안시', '동남구', '서북구', '공주시', '보령시', '아산시', '서산시', '논산시', '계룡시', '당진시', '금산군', '부여군', '서천군',
                           '청양군', '홍성군', '예산군', '태안군', '전주시', '완산구', '덕진구', '군산시', '익산시', '정읍시', '남원시', '김제시', '완주군', '진안군',
                           '무주군', '장수군', '임실군', '순창군', '고창군', '부안군', '목포시', '여수시', '순천시', '나주시', '광양시', '담양군', '곡성군', '구례군',
                           '고흥군', '보성군', '화순군', '장흥군', '강진군', '해남군', '영암군', '무안군', '함평군', '영광군', '장성군', '완도군', '진도군', '신안군',
                           '포항시', '경주시', '김천시', '안동시', '구미시', '영주시', '영천시', '상주시', '문경시', '경산시', '군위군', '의성군', '청송군', '영양군',
                           '영덕군', '청도군', '고령군', '성주군', '칠곡군', '예천군', '봉화군', '울진군', '울릉군', '창원시', '의창구', '성산구', '마산합포구',
                           '마산회원구', '진해구', '진주시', '통영시', '사천시', '김해시', '밀양시', '거제시', '양산시', '의령군', '함안군', '창녕군', '남해군', '하동군', '산청군',
                           '함양군', '거창군', '합천군', '제주시', '서귀포시']

        #'제주시'는 위계가 다르기 때문에 '제주'는 변환하지 않는다!!!
        sido_substitute_dict = [{"abbr": "서울시", "org": "서울특별시"}, {"abbr": "서울", "org": "서울특별시"},
                           {"abbr": "인천시", "org": "인천광역시"}, {"abbr": "인천", "org": "인천광역시"},
                           {"abbr": "부산시", "org": "부산광역시"}, {"abbr": "부산", "org": "부산광역시"},
                           {"abbr": "대구시", "org": "대구광역시"}, {"abbr": "대구", "org": "대구광역시"},
                           {"abbr": "광주시", "org": "광주광역시"}, {"abbr": "광주", "org": "광주광역시"},
                           {"abbr": "대전시", "org": "대전광역시"}, {"abbr": "대전", "org": "대전광역시"},
                           {"abbr": "경기", "org": "경기도"}, {"abbr": "강원", "org": "강원도"},
                           {"abbr": "충북", "org": "충청북도"}, {"abbr": "충남", "org": "충청남도"},
                           {"abbr": "전북", "org": "전라북도"}, {"abbr": "전남", "org": "전라남도"},
                           {"abbr": "경북", "org": "경상북도"}, {"abbr": "경남", "org": "경상남도"},
                           {"abbr": "세종시", "org": "세종특별자치시"}, {"abbr": "세종", "org": "세종특별자치시"},
                           {"abbr": "제주도", "org": "제주특별자치도"}]

        # first_word : 대한민국 17개 특별시, 광역시, 도, 자치도, 자치시가 저장된 변수(ex : 서울특별시)
        first_word = ""

        # first_half_word : 대한민국 약 200개에 해당하는 시, 군, 구가 저장된 변수(ex : 광진구)
        first_half_word = ""

        # second_word : (~길, ~로)를 저장하는 변수(ex : 보라매로5길)
        second_word = ""

        # address_number : 번지를 저장하는 변수(ex : 75, 120-1)
        address_number = ""

        # word_list : one_line(서울특별시 동작구 보라매로5길 20)에 저장되어 있는 Str을 띄어쓰기한칸을 기준으로 split하여 list형태로 저장
        # 만약 one_word(split된 단어)가 빈값이라면 저장하지 않는다.

        # ex : 서울특별시 동작구 보라매로5길 20 ---> ['서울특별시', '동작구', '보라매로5길', '20']
        word_list = [one_word for one_word in one_line.split(" ") if one_word is not ""]
        # 맨 앞 아이템에 대해서 축약형 전처리를 수행 (2021.01.15)
        # 맨 앞이 아니어도 시행은 가능하나, 우려되는 케이스로 인해 하지 않는 것으로 한다. Ex) 서울시청로 265를 서울 시청로 265로 적는 오기의 경우 문제의 소지가 존재.
        word_list[0] = sido_abbr_substitute(word_list[0])
        # 축약형/혹은 완전형 시도에 시군구가 결합된 경우(부산사하구 or 부산광역시사하구)
        # 첫 번째가 아닌 경우도 대응토록 한다.
        for word in word_list:
            position = word_list.index(word)
            for sgg in small_unit_list:
                # 그냥 묶은 다음 비교함. 컴퓨팅 파워를 믿자
                for sd in unit_list:
                    sgg_sd_sum = sd + sgg
                    if word == sgg_sd_sum:
                        word_list[position] = sd
                        word_list.insert(position + 1, sgg)
                for sd in sido_substitute_dict:
                    sgg_sd_sum = sd['abbr'] + sgg
                    if word == sgg_sd_sum:
                        word_list[position] = sd['org']
                        word_list.insert(position + 1, sgg)
        print(word_list)



        # final_word_list : word_list의 끝 단어만을 저장한 list
        # ex : ['서울특별시', '동작구', '보라매로5길', '20'] ---> ['시', '구', '길', '20']
        final_word_list = [word[-1] for word in word_list]

        # final_word_list_only_letter : word_list중에서 숫자가 포함되어 있는 것을 모두 삭제하고 끝 단어만을 저장한 list
        # word_list로 for문을 돌면서, final_word_list_only_letter값을 확정
        # ex : ['서울특별시', '동작구', '보라매로5길', '20'] ---> ['시', '구', '길']
        final_word_list_only_letter = []
        for text in word_list:
            text = text.replace("-", "")
            if len(''.join([i for i in text if not i.isdigit()])) > 0:
                final_word_list_only_letter.append(''.join([i for i in text if not i.isdigit()])[-1])

        # 끝단어로 "길"이 존재하는경우
        if "길" in final_word_list_only_letter:
            # word_list(['서울특별시', '동작구', '보라매로5길', '20'])를 한단어씩 확인
            # text ---> '서울특별시', '동작구', '보라매로5길', '20'
            for idx, text in enumerate(word_list):
                # 1. 대한민국 17개 특별시, 광역시, 도, 자치도, 자치시에 해당하는 단어가 존재하면 first_word에 저장
                if text in unit_list:
                    first_word = text

                if text in small_unit_list:
                    first_half_word = text

                # 3. 끝단어가 "길"로 끝나는것이 있는가? ---> 끝단어가 "로"로 끝나는 것을 포함하는가?
                if text[-1] == "길":  # ~길 구조검사
                    # 3-1. # 길은 있는데 로는 없는 경우.
                    if "로" not in final_word_list:
                        second_word = text
                        address_number = re.search(r"[\d-]+", word_list[idx + 1]).group()
                    # 3-2. # 길과 로가 같이 있는 경우.
                    else:
                        #~로 ~번 ~길과 같이 오표기한 경우
                        if "번" in final_word_list_only_letter and final_word_list_only_letter.index("번") == final_word_list_only_letter.index("길") - 1 and final_word_list_only_letter.index("로") == final_word_list_only_letter.index("번") - 1:
                            second_word = word_list[idx - 2] + word_list[idx - 1] + text
                            address_number = re.search(r"[\d-]+", word_list[idx + 1]).group()
                        else:
                            second_word = word_list[idx - 1] + text
                            address_number = re.search(r"[\d-]+", word_list[idx + 1]).group()
                # 예외 : ~길84와 같이 오표기한 경우
                elif bool(re.search(r"길[\d-]+", text)):
                    # 이 경우에도 ~로 ~번 ~길 오표기가 존재할 수 있음.
                    if "번" in final_word_list_only_letter and final_word_list_only_letter.index("번") == final_word_list_only_letter.index("길") - 1 and final_word_list_only_letter.index("로") == final_word_list_only_letter.index("번") - 1:
                        second_word = word_list[idx - 2] + word_list[idx - 1] + text
                    # 바꿔도 오표기는 여전하니 이것을 수정해 줌.(2021.01.15)
                    #else:
                    #   second_word = text
                    text = text.replace("길", "길 ")
                    second_word = text.split(" ")[0]
                    address_number = text.split(" ")[1]

        # 끝단어로 "길"이 존재하지 않으면서 "로"가 존재하는경우
        elif "로" in final_word_list_only_letter:
            # word_list(['강원도', '강릉시', '경강로', '2007'])를 한단어씩 확인
            # text ---> '강원도', '강릉시', '경강로', '2007'
            for idx, text in enumerate(word_list):
                # 1. 대한민국 17개 특별시, 광역시, 도, 자치도, 자치시에 해당하는 단어가 존재하면 first_word에 저장
                if text in unit_list:
                    first_word = text

                if text in small_unit_list:
                    first_half_word = text

                # 3-1. 끝단어가 "로"로 끝나는것이 있으면 바로 저장
                if text[-1] == "로":
                    second_word = text
                    address_number = re.search(r"[\d-]+", word_list[idx + 1]).group()  # 앞과 같은 숫자찾기
                # 3-2. ~로84와 같이 오표기한 경우
                elif bool(re.search(r"로[\d-]+", text)):
                    # 바꿔도 오표기는 여전하니 이것을 수정해 줌.(2021.01.15)
                    #second_word = text
                    text = text.replace("로", "로 ")
                    second_word = text.split(" ")[0]
                    address_number = text.split(" ")[1]
        # 끝단어로 "길" "로" 모두없는 경우 에러로 처리
        else:
            return "Error"

        # 후처리: 스페이싱 없이 그냥 죄다 붙여 쓴 경우!
        # 조건은?
        # 1번 2번 단어가 싹 없음(시도X 시군구X) : 의심하는 것-> '부산사하구하신번영로15번길10' 의 사례
        if first_half_word == "" and first_word == "":
            alter_checker = False
            jeju_checker = False
            # 시도목록의 아이템으로 시작하는지 검토해서 있으면 바꿈
            for item in unit_list:
                if second_word[0:len(item)] == item:
                    second_word = second_word[len(item):]
                    first_word = item
                    alter_checker = True

            if alter_checker == False:
                # 시도축약형 목록의 아이템으로 시작하는지 검토해서 있으면 바꿈 (second_word에서 제거한 뒤 first_word를 추가)
                # 시도목록의 아이템이 변경되었을 경우엔 이 로직을 검토하지 않는다 (부산광역시 부산동구~와 같은 경우를 방지하기 위함)
                # 이 경우엔 문제의 소지가 있음. 가령 앞 주소가 아예 없고 경기대로 312 만 존재하는 경우는...? 물론 구별성이 떨어지지만 이게 unique한 경우도 있지 않을까?
                for item in sido_substitute_dict:
                    if second_word[0:len(item['abbr'])] == item['abbr']:
                        second_word = second_word[len(item['abbr']):]
                        first_word = item['org']

                # 제주는 시도/시군구에 전부 있으므로 그냥 놓아 둔다...여기까지 왔으면 '제주특별자치도' '제주도' 둘 다 아님.
                if second_word[0:2] == "제주" and second_word[2] != "대":  # 제주대학로, 제주대학로4길에 대한 예외케이스 설정.
                    if second_word[2] == '시':  # 제주시로 올바르게 적은 경우. 그러나 2번 단어임.
                        second_word = second_word[3:]
                        first_half_word = "제주시"
                        jeju_checker = True
                    else:
                        second_word = second_word[2:]
                        first_word = "제주"
                        jeju_checker = True
            # 이 시점에서 first word 탐색 시도는 종료. first_half_word가 없는 상태임! (시군구)
            # second word를 다시 검토해서 부산하신번영로15번길10과 부산사하구하신번영로15번길10을 구분할 수 있도록 함.

            # 시군구목록의 아이템으로 시작하는지 검토해서 있으면 바꿈
            for item in small_unit_list:
                if second_word[0:len(item)] == item:
                    second_word = second_word[len(item):]
                    first_half_word = item
            if second_word[0:2] == "제주" and second_word[2] != "대":  # 제주대학로, 제주대학로4길에 대한 예외케이스 설정.
                if jeju_checker == True:
                    # '제주'가 2번 있는 경우! 이건 제주도 제주시~임.
                    first_word = "제주특별자치도"
                    first_half_word = "제주시"
                    second_word = second_word[2:]
                else:
                    # '제주'가 처음 나온 경우. 이건 모르니까 '제주'로 놓는다
                    second_word = second_word[2:]
                    first_half_word = "제주"
                    jeju_checker = True
                # 원래대로면 제주 서귀포시까지는 체크를 해야 하는데, 제주 서귀포시라고 입력해도 행안부 검색은 될 듯. 나중에 문제가 된다면 수정할 것.
                print(first_word)
                print(first_half_word)
                print(second_word)

        # 2번째 단어(시군구)만 없는 경우. 1번째 단어만 없는 경우는 있을 수 없다! (있더라도 도로명에 붙은 것이 아니라 시도-시군구가 붙은 것. 가령- 부산사하구 하신번영로15번길10)
        # 의심하는 사례 -> '부산광역시 사하구하신번영로15번길10'
        elif first_half_word == "":
            # 시군구목록의 아이템으로 시작하는지 검토해서 있으면 바꿈
            for item in small_unit_list:
                if second_word[0:len(item)] == item:
                    second_word = second_word[len(item):]
                    first_half_word = item

            # 제주시 특이케이스 분기. 이 경우는 시도가 나왔으니까 무조건 제주시임. 근데 제주시면 어차피 제주도 아닌가??
            # 검색결과 제주시로 시작하는 길 이름은 없음 ex)제주시청로
            if second_word[0:2] == "제주" and second_word[2] != "대":  # 제주대학로, 제주대학로4길에 대한 예외케이스 설정.
                first_half_word = "제주시"
                second_word = second_word[2:]

        # elif first_word == "":  #여기의 반례는 현재로서는 시도 축약구분이 안되는 제주 서귀포시 ~~밖에 없음.... 서귀포시로 식별 가능하므로 지움.

        #강변안길 100-과 같은 주소오기 처리(하이픈으로 끝나는 경우)
        #*주의* 이 알고리즘은 기타 도로주소 오브젝트에 -가 사용되지 않는 것을 전제로 함. 가령, 춘천-시 같은게 있으면 지워짐!
        ##덧) 하지만 그런 경우는 춘천-시 : 춘천시 로 변환되는데 좋은 것이 아닌지?
        full_address = first_word + " " + first_half_word + " " + second_word + " " + address_number
        #print("시도명 : " + first_word)
        #print("시군구명 : " + first_half_word)
        #print("도로명 : " + second_word)
        #print("번지 : " + address_number)
        #변수에 저장하고, 전처리를 했음에도 불구하고 대시 뒤에 공백이 생기는 경우가 있음. 이걸 지움. 빌딩 케이스는 "경상북도 영주시 대학로327-"
        if bool(re.search("[-][ ]+", full_address)):
            full_address = full_address.replace(re.search("[-][ ]+", full_address).group(), "-")
            print('-(대시) 뒤 공백 처리됨')
        #맨 뒤가 -로 끝나면 없앰. -오류의 여지가 있지 않나? pop같은 걸로 바꾸는 것을 고려할 것.
        if full_address[-1] == "-":
            full_address = full_address.replace("-","")

        return full_address
    # 최종 에러 처리
    except:
        print("Error")
        return "Error"

