version1
將資料分成訓練集2019/01到2019/10，結果是隔年2020/01到2020/10，
然後測試集為2019/11到2020/01結果為2020/11到2021/01，
其中台電下載資料csv僅提供到2020/10，我去網站上查到2020/11~2021/01的手動keyin成CSV檔，
然後再利用SGDRegressor用2020/03去預測2021/03，擷取2021/03/23~2021/03/29輸出結果。