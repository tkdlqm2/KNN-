import operator
import csv
from collections import Counter
from linear_algebra import distance

def select_stockname(stock_data, stockname):
    tempArr = []
    for i in range(len(stock_data)):
        if stock_data[i][1] == stockname:
            tempArr.append(stock_data[i])
    return tempArr

# jewoo 5/25
def find_appropriate_udnd_stockname(stock_data):
    unsigned_udnd = 0   # udnd = 1
    signed_udnd = 0  # udnd = -1
    appropriate_udnd_stockname_list = []
    # 마지막 인덱스 전까지 for문
    for i in range(len(stock_data)-1):
        if stock_data[i][1] == stock_data[i+1][1]:
            if stock_data[i][12] == 1:
                unsigned_udnd += 1
            if stock_data[i][12] == -1:
                signed_udnd += 1
        else:
            if unsigned_udnd >= 20 & signed_udnd >= 20:
                appropriate_udnd_stockname_list.append(stock_data[i][1])
                signed_udnd = 0
                unsigned_udnd = 0
            else:
                signed_udnd = 0
                unsigned_udnd = 0
    # 마지막 인덱스 계산
    if stock_data[len(stock_data)-1][12] == 1:
        unsigned_udnd += 1
    if stock_data[len(stock_data)-1][12] == -1:
        signed_udnd += 1
    if unsigned_udnd >= 20 & signed_udnd >= 20:
        appropriate_udnd_stockname_list.append(stock_data[len(stock_data)-1][1])
    return appropriate_udnd_stockname_list


def get_cv_diff_value(stock_data):
    stock_data[0].append(0)
    for i in range(1, len(stock_data)):
        stock_data[i].append(stock_data[i][6] - stock_data[i-1][6]) # 종가변화량 구해서 인덱스8번쨰에 append

def get_cv_diff_rate(stock_data):
    stock_data[0].append(0)
    for i in range(1, len(stock_data)):
        try:
            # 종가변화율 = (오늘close - 어제close / 어제close) * 100
            stock_data[i].append(round(((stock_data[i][6] - stock_data[i-1][6]) / stock_data[i-1][6]) * 100, 2)) # 종가변화율 구해서 인덱스9번째에 append, 소수점 2번 째 자리까지만
        except ZeroDivisionError as e:
            stock_data[i].append(0)  # 0으로 나누기 불가. 근데 엑셀에는 '0으로 나누기 불가'라고 쓰기 그러니까 0으로 저장

def get_cv_maN_value(N, stock_data):
    for i in range(N-1):
        stock_data[i].append(0)
    for i in range(N-1, len(stock_data)):
        sum_close_value = 0
        for j in range(N):
            sum_close_value += stock_data[i-j][6]
        stock_data[i].append(sum_close_value / N)   # 이동평균 구해서 인덱스10번쨰에 append

def get_cv_maN_rate(stock_data):
    stock_data[0].append(0)
    for i in range(1, len(stock_data)):
        try:
            # 이동평균변화율 = (오늘maN - 어제maN / 어제maN) * 100
            stock_data[i].append(
                round( ((stock_data[i][10] - stock_data[i-1][10]) / stock_data[i-1][10]) * 100, 2))
            # 이동편균변화율 구해서 인덱스11번쨰에 append, 소수점 2번 째 자리까지만
        except ZeroDivisionError as e:
            stock_data[i].append(0) # 0으로 나누기 불가. 근데 엑셀에는 '0으로 나누기 불가'라고 쓰기 그러니까 0으로 저장

def get_ud_Nd(N, stock_data):
    for i in range(len(stock_data)):
        count = 0
        if i < N-1:
            stock_data[i].append('비교불가') # 비교불가
        elif i > N-1:
            for j in range(N):
                if stock_data[i-j-1][6] - stock_data[i-j][6] < 0:
                    count += 1
                elif stock_data[i-j-1][6] - stock_data[i-j][6] > 0:
                    count -= 1
            if count == N:
                stock_data[i-1].append(1)
            elif count == -N:
                stock_data[i-1].append(-1)
            else:
                stock_data[i-1].append(0)
    stock_data[len(stock_data)-1].append('비교불가') # 비교불가

def get_cvNd_diff_rate(N, stock_data):
    for i in range(len(stock_data)):
        if i < N - 2:
            stock_data[i].append('비교불가') # 비교불가
        elif i > N - 2:
            try:
                stock_data[i-1].append( round( ((stock_data[i][6] - stock_data[i-N+1][6]) / stock_data[i-N+1][6]) *100, 2) )
            except ZeroDivisionError as e:
                stock_data[i-1].append(0)  # 0으로 나누기 불가. 근데 엑셀에는 '0으로 나누기 불가'라고 쓰기 그러니까 0으로 저장
    stock_data[len(stock_data)-1].append('비교불가') # 비교불가

def get_vv_diff_value(stock_data):
    stock_data[0].append(0)
    for i in range(1, len(stock_data)):
        stock_data[i].append(stock_data[i][7] - stock_data[i-1][7]) # 거래량변화량 구해서 인덱스14번쨰에 append

def get_vv_diff_rate(stock_data):
    stock_data[0].append(0)
    for i in range(1, len(stock_data)):
        try:
            # 거래량변화율 = (오늘voulume - 어제voulume / 어제voulume) * 100
            stock_data[i].append(round(((stock_data[i][7] - stock_data[i-1][7]) / stock_data[i-1][7]) * 100, 2)) # 거래량변화율 구해서 인덱스15번쨰에 append, 소수점 2번 째 자리까지만
        except ZeroDivisionError as e:
            stock_data[i].append(0)  # 0으로 나누기 불가. 근데 엑셀에는 '0으로 나누기 불가'라고 쓰기 그러니까 0으로 저장

def get_vv_maN_value(N, stock_data):
    for i in range(N-1):
        stock_data[i].append(0)
    for i in range(N-1, len(stock_data)):
        sum_volume_value = 0
        for j in range(N):
            sum_volume_value += stock_data[i-j][7]
        stock_data[i].append(sum_volume_value / N)   # 이동평균 구해서 인덱스16번쨰에 append

def get_vv_maN_rate(stock_data):
    stock_data[0].append(0)
    for i in range(1, len(stock_data)):
        try:
            # 이동평균변화율 = (오늘maN - 어제maN / 어제maN) * 100
            stock_data[i].append(
                round( ((stock_data[i][16] - stock_data[i-1][16]) / stock_data[i-1][16]) * 100, 2))
            # 이동편균변화율 구해서 인덱스17번쨰에 append, 소수점 2번 째 자리까지만
        except ZeroDivisionError as e:
            stock_data[i].append(0)

def save_data_to_csvfile(filename, lines):
    wf = open(filename,'w', newline='')
    csv_writer = csv.writer(wf)
    for line in lines:
        csv_writer.writerow(line)

def majority_vote(labels):
    """assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner  # unique winner, so return it
    else:
        return majority_vote(labels[:-1])  # try again without the farthest


def knn_classify(k, labeled_points, new_point):
    """each labeled point should be a pair (point, label)"""

    # order the labeled points from nearest to farthest
    # print(labeled_points, new_point)
    by_distance = sorted(labeled_points,
                         key=lambda point_label: distance(point_label[0], new_point))

    # find the labels for the k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # and let them vote
    return majority_vote(k_nearest_labels)

def make_labels(stock_data, independent_values_indexlist): # param2: 독립변수 '인덱스' 리스트[]
    label = []
    for i in range(len(stock_data)):
        independent_values = []
        for j in independent_values_indexlist:  # 독립변수'값'리스트[] 만들고
            independent_values.append(stock_data[i][j])
        label.append((independent_values, stock_data[i][12]))
    return label

def filtering_not_number(stock_data):
    filtered_stock = []
    for i in range(len(stock_data)):
        if stock_data[i][12] == '비교불가': continue
        if stock_data[i][13] == '비교불가': continue
        filtered_stock.append(stock_data[i])
    return filtered_stock

def divide_training_and_test_label(stock_data,divisionNumber):
    label1 = [] # 트레이닝 테이터 라벨
    label2 = [] # 테스트 데이터 라벨
    for i in range(0,divisionNumber):
        label1.append(stock_data[i])
    for i in range(divisionNumber,len(stock_data)):
        label2.append(stock_data[i])
    return label1,label2

def training_data(labeled_stock_data):
    accuracy = []
    for k in range(3,8): # 테스트해볼 k값 범위 3~7
        num_correct = 0

        for independent_variables, actual_ud_Nd in labeled_stock_data:
            other_stocks = [other_stock
                            for other_stock in labeled_stock_data
                            if other_stock != (independent_variables, actual_ud_Nd)]
            predicted_ud_Nd = knn_classify(k, other_stocks, independent_variables)

            if predicted_ud_Nd == actual_ud_Nd:
                num_correct += 1

        accuracy.append((k, round((num_correct / len(labeled_stock_data) * 100), 2)))
    return accuracy

def get_match_result(result1, result2):
    match_result = []
    k_list = []
    training_accuracy = []
    test_accuracy = []
    for i in range(len(result2)):
        k_list.append(result2[i][0])
        training_accuracy.append(result1[i][1])
        test_accuracy.append(result2[i][1])
    zipped_k_training_test = list(zip(k_list, training_accuracy, test_accuracy))
    for z in range(len(zipped_k_training_test)):
        # if int(zipped_k_training_test[z][1]) > 70: # 트레이닝데이터 정확도 70% 이상이고,
        if int(zipped_k_training_test[z][1]) - 5 < int(zipped_k_training_test[z][2]) < int(zipped_k_training_test[z][1]) + 5:  # 트레이닝데이터와 테스트데이터의 정확도 오차가 +-5%이면 (over fitting 방지)
            match_result.append((zipped_k_training_test[z][0], zipped_k_training_test[z][2]))   # 그 때의 k값과 test데이터의 정확도 저장
    print("matchInGet:",match_result)
    return match_result

def for_get_match_result(stock_data):
    matched_result = []
    iterable_columns = [3,4,5,6,7,8,9,10,11,13,14,15,16,17]
    filtered_stock = filtering_not_number(stock_data)

    for i in iterable_columns:  # 시작컬럼 for문
        index_pointer = iterable_columns.index(i) + 1
        while index_pointer < len(iterable_columns) + 1:  # 컬럼조합 while문
            column_indexlist = [i]  # 시작컬럼
            for j in range(index_pointer - 1, len(iterable_columns)):
                if j == iterable_columns.index(i): index_pointer += 1
                else: column_indexlist = column_indexlist + [iterable_columns[j]]  # 시작, 마지막 컬럼 외에는 모든 컬럼조합하기
                print(column_indexlist)
                labeled_target_stock = make_labels(filtered_stock, column_indexlist) # labeled_target_stock = ([독립변수들], 종송변수udnd) - 모든 filter_stock_data에 대해 해당 ([독립변수리스트], 종속변수udnd)로 라벨링한다.
                len_training_data = int(len(labeled_target_stock) * 0.7)
                training_label, test_label = divide_training_and_test_label(labeled_target_stock, len_training_data)  # 트레이닝, 테스트 7:3으로 나누기
                result_accuracy_training = training_data(training_label)  # result_accruacy = (k, 그 때의 정확도), (k, 그 때의 정확도), (k, 그 때의 정확도)...
                result_accuracy_test = training_data(test_label)  # result_accruacy = (k, 그 때의 정확도), (k, 그 때의 정확도), (k, 그 때의 정확도)...
                result_appropriate_test_accuracy = get_match_result(result_accuracy_training, result_accuracy_test) # result_appropriate_test_accuracy: (k, 그때의 정확도), (k, 그때의 정확도), (k, 그때의 정확도)...
                if result_appropriate_test_accuracy != []:  # 값이 없는 경우 패스
                    for k in range(len(result_appropriate_test_accuracy)):
                        matched_result.append((column_indexlist, result_appropriate_test_accuracy[k][0], result_appropriate_test_accuracy[k][1]))
            index_pointer += 1

    return matched_result

if __name__ == "__main__":
    stock_column = []
    target_cloumn = []
    target_stock = []
    final_N = 3
    final_stockname = '옐로페이'
    # 전체 데이터셋을 위한 변수
    all_stock_data = []
    udnd20_stockname_list = []
    labeled_target_stock_list = []
    training_label_list = []
    test_label_list = []
    resultlist_training = []
    resultlist_test = []

    with open("stock_history.csv", "r") as fread:
        line = fread.readlines()
        stock_column = line[0].split(',')[:8]
        for i in range(1, len(line)):
            all_stock_data.append(line[i].split(',')[:8])
    for i in range(len(all_stock_data)):    # 엑셀에서 가져온 데이터를 날짜/종목명 빼고 모두 int형으로 변환
        for j in range(2, 8):
            all_stock_data[i][j] = int(all_stock_data[i][j])
    target_stock = select_stockname(all_stock_data, final_stockname)
    target_stock = sorted(target_stock, key=operator.itemgetter(0))
    target_cloumn = stock_column;
    get_cv_diff_value(target_stock)
    target_cloumn.append('cv_diif_value')
    get_cv_diff_rate(target_stock)
    target_cloumn.append('cv_diif_rate')
    get_cv_maN_value(final_N, target_stock)
    target_cloumn.append('cv_maN_value')
    get_cv_maN_rate(target_stock)
    target_cloumn.append('cv_maN_rate')
    get_ud_Nd(final_N, target_stock)
    target_cloumn.append('ud_Nd')
    get_cvNd_diff_rate(final_N, target_stock)
    target_cloumn.append('cvNd_diff_rate')
    get_vv_diff_value(target_stock)
    target_cloumn.append('vv_diif_value')
    get_vv_diff_rate(target_stock)
    target_cloumn.append('vv_diif_rate')
    get_vv_maN_value(final_N, target_stock)
    target_cloumn.append('vv_maN_value')
    get_vv_maN_rate(target_stock)
    target_cloumn.append('vv_maN_rate')

    good_result = for_get_match_result(target_stock)

    # 파일저장
    good_result_column = ('독립변수', 'k', '테스트정확도', 'N='+str(final_N), final_stockname)
    good_result.insert(0, good_result_column)
    save_data_to_csvfile('stock_history_K.csv', good_result)
    target_stock.insert(0, target_cloumn)
    save_data_to_csvfile('stock_history_added.csv', target_stock)