# RoadDamageDetector 
道路の損傷箇所と損傷の種類を予測するニューラルネットワーク.
Pytorch 1.0を使用している．ニューラルネットワークの構成はYOLOを参考にしており，損失関数についても参考にしている．また，ベースとしてvgg16の学習済みの層を前段に使用している.


## Install 
```
conda install pytorch torchvision -c pytorch
conda install opencv
```

## 準備
1. config.pyを書き換える
    ```python
    # 学習データがあるフォルダに設定 
    dirname_trainimage = 'input/train'
    # テスト用の画像があるフォルダに設定
    dirname_testimage = 'input/test'
    # テスト用の画像の予測結果の画像を出力するフォルダを設定 
    dirname_testimage_predict = 'results/test'
    # モデルパラメータの保存先
    fn_model = 'results/model.pt'
    ```

## 学習
`python train.py`
    

## 推論
`python predict.py`
計算が終了したら`results/answer.xml`に予測された損傷boxを記述したファイルが出力される.