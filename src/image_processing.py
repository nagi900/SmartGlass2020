#画像処理のモジュール

import math

#初めから極座標で画像の位置を決めると、二つめのディスプレイで表示するとき面倒なので、これで変換
def output_img_tranceformation( #引き数は全て必須
        OIT_img,
        OIT_x, OIT_y, OIT_z,
        vertex_distance,
        display_width,
        eye_coordinate, #眉間を0としたときの目のx座標
        ):

    #高さVERTEX_DISTANCE、底辺DISPLAY_WIDTHの二等辺三角形を
    #半径VERTEX_DISTANCE、位相arctan{DISPLAY_WIDTH/ (VERTEX_DISTANCE/2) }
    #の扇とする
    
    #ディスプレイで表示できる第一証言の最大角度をもとめる 単位はラジアン
    max_display_angle_x = math.atan(display_width / (2*vertex_distance))
 
    #オブジェクトの座標を目の位置に合わせてずらす
    OIT_x = OIT_x + eye_coordinate

    #画面に表示するx座標=水平方向角度　y座標=垂直方向角度 となる
    object_coordinate_x = math.atan(OIT_x / OIT_z)
    object_coordinate_y = math.atan(OIT_y / OIT_z)

    return object_coordinate_x,object_coordinate_y

