if __name__ == '__main__':
    import csv
    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, SGDRegressor
    # print(csv.__version__)
    # print(np.__version__)
    # print(pd.__version__)
    # print(sklearn.__version__)

    #開啟 CSV 檔案
    # with open('raw data.csv', newline='',encoding="utf-8") as csvfile:
        #讀取 CSV 檔案內容
        # rows = csv.reader(csvfile)
        # with open('X_train.csv', 'w', newline='',encoding="utf-8") as csvfile1:   #X_train
        #     # 建立 CSV 檔寫入器
        #     writer = csv.writer(csvfile1)
        #     k=0
        #     for row in rows:
        #         # 寫入一列資料
        #         if k == 305:
        #             break
        #         if k > 0:
        #             writer.writerow(row[1:7])
        #         k = k + 1
        # with open('Y_train.csv', 'w', newline='',encoding="utf-8") as csvfile2:     #Y_train
        #     writer = csv.writer(csvfile2)
        #     k = 0
        #     for row in rows:
        #         if k > 365:
        #             if k != 425:    #2020/02/29 del
        #                 writer.writerow(row[3:4])
        #         k = k + 1
        # with open('X_test.csv', 'w', newline='',encoding="utf-8") as csvfile2:     #X_test
        #     writer = csv.writer(csvfile2)
        #     k = 0
        #     for row in rows:
        #         if k > 304:
        #             writer.writerow(row[1:7])
        #         k = k + 1
        #         if k > 396:
        #             break
    # with open('2020.11.01_2021.01.31_data.csv', newline='',encoding="utf-8") as csvfile:    #此資料為台電提供20/11/01~21/01/31手動輸入之資料
    #     #讀取 CSV 檔案內容
    #     rows = csv.reader(csvfile)
    #     with open('Y_test.csv', 'w', newline='',encoding="utf-8") as csvfile1:   #Y_test
    #         # 建立 CSV 檔寫入器
    #         writer = csv.writer(csvfile1)
    #         for row in rows:
    #             # 寫入一列資料
    #             writer.writerow(row[1:2])
    # with open('Y_test.csv', newline='',encoding="utf-8") as p:
    #     rows = csv.reader(p)
    #     for row in rows :
    #         print(row)
    # with open('raw data.csv', newline='',encoding="utf-8") as csvfile:  #2020/03 predict 2021/03
    #     #讀取 CSV 檔案內容
    #     rows = csv.reader(csvfile)
    #     with open('predict_data.csv', 'w', newline='',encoding="utf-8") as csvfile:
    #         writer = csv.writer(csvfile)
    #         k = 0
    #         for row in rows:
    #             if k > 447:
    #                 writer.writerow(row[1:7])
    #             k = k + 1
    #             if(k>454):
    #                 break
    #將資料轉為 pandas DataFrame
    with open('X_train.csv', newline='',encoding="utf-8") as csvfile:
        rows = csv.reader(csvfile)
        A = []
        for row in rows:
            A.append(row)
        X_train = pd.DataFrame(A)
    with open('Y_train.csv', newline='',encoding="utf-8") as csvfile:
        rows = csv.reader(csvfile)
        A = []
        for row in rows:
            A.append(row)
        Y_train = pd.DataFrame(A)
    with open('X_test.csv', newline='',encoding="utf-8") as csvfile:
        rows = csv.reader(csvfile)
        A = []
        for row in rows:
            A.append(row)
        X_test = pd.DataFrame(A)
    with open('Y_test.csv', newline='',encoding="utf-8") as csvfile:
        rows = csv.reader(csvfile)
        A = []
        for row in rows:
            A.append(row)
        Y_test = A
    # print(Y_train)
    #標準化
    ss_x = StandardScaler()
    X_train = ss_x.fit_transform(X_train)
    X_test = ss_x.transform(X_test)
    lr = LinearRegression()
    # 訓練
    lr.fit(X_train, Y_train)
    # 預測 保存預測結果
    lr_y_predict = lr.predict(X_test)
    # 初始化SGDRRegressor隨機梯度回歸模型
    sgdr = SGDRegressor()
    # 訓練
    sgdr.fit(X_train, Y_train)
    # 預測 保存預測結果
    sgdr_y_predict = sgdr.predict(X_test)
    #結果
    # print(lr_y_predict)
    # print(sgdr_y_predict)
    sum_lr = 0
    for i in range(len(lr_y_predict)):
        sum_lr = sum_lr + (int(lr_y_predict[i][0]) - int(Y_test[i][0]))**2
    sum_lr = sum_lr / len(lr_y_predict)
    sum_lr = sum_lr**0.5

    sum_sgdr = 0
    for i in range(len(sgdr_y_predict)):
        sum_sgdr = sum_sgdr + (int(sgdr_y_predict[i]) - int(Y_test[i][0]))**2
    sum_sgdr = sum_sgdr / len(sgdr_y_predict)
    sum_sgdr = sum_sgdr**0.5
    print('lr:',sum_lr)
    print('sgdr:',sum_sgdr)
    with open('predict_data.csv', newline='',encoding="utf-8") as csvfile:
        rows = csv.reader(csvfile)
        A = []
        for row in rows:
            A.append(row)
        predict_data = pd.DataFrame(A)
    predict_data = ss_x.transform(predict_data)
    sgdr_result_predict = sgdr.predict(predict_data) 
    #print(sgdr_result_predict)

    k = 20210323
    A = []
    for i in range(len(sgdr_result_predict)):
        A.append([k,sgdr_result_predict[i]])
        k = k+1
    #print(pd.DataFrame(A))
    with open('submission.csv', 'w', newline='',encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(len(sgdr_result_predict)):
                writer.writerow(A[i])