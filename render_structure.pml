# コマンドライン引数からCIFファイルを読み込む
#load $1
load output/yuan_shuffle_1/yuan_shuffle_1_model.cif 
# 表示形式を設定
show cartoon
color green, all

# 背景色を設定
bg_color white

# ズームと視点の調整
zoom

# 画像を保存
png output_image.png, width=1080, height=1080, dpi=300
